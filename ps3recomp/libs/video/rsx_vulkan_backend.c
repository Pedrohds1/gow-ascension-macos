/*
 * ps3recomp - Vulkan RSX Backend. See rsx_vulkan_backend.h.
 *
 * The file, in order:
 *   loader and device      libvulkan by dlopen, Vulkan 1.0 entry points only;
 *                          device choice, BC textures when the device has them
 *   memory, images, submit helpers
 *   vtable fallback path   the null backend's fixed-function contract:
 *                          position = attribute 0, flat colour = attribute 3
 *                          of each triangle's first vertex, texcoord0 =
 *                          attribute 8, fetched and expanded through the
 *                          shared rsx_fetch_attrib(); unit 0 decoded through
 *                          rsx_texture_decode(); depth test, no blend/stencil
 *   vtable guest programs  the same path running the decompilers' HLSL, with
 *                          whole mip chains and per-unit samplers
 *   optional window        SDL2 + swapchain; a present blits the frame into it
 *   draw engine backend    rsx_draw_backend: targets, clears, present and
 *                          readback (E1), pipelines and draws (E2), textures,
 *                          surface views and samplers (E3), depth snapshots
 *                          (E4) -- the stage names in the commit history
 *   public entry points    init picks the path, shutdown undoes everything
 *
 * Deliberately synchronous: every clear, draw and present is one one-shot
 * command buffer, submitted and waited on. Slow and trivially correct; it is
 * what lets the engine's submit_and_wait and staging reuse cost nothing, and
 * batching is the first thing to change once correctness is settled.
 *
 * Images stay in VK_IMAGE_LAYOUT_GENERAL for their whole life: legal for
 * transfers, sampling and as attachments, so there is no layout bookkeeping.
 *
 * Shaders: libs/video/vulkan/rsx_vk_fallback.{vert,frag}, embedded as SPIR-V.
 * Regenerate after editing them with:
 *   glslangValidator -V --target-env vulkan1.0 --vn rsx_vk_fallback_vert_spv \
 *       -o rsx_vk_fallback_vert.spv.h rsx_vk_fallback.vert
 *   glslangValidator -V --target-env vulkan1.0 --vn rsx_vk_fallback_frag_spv \
 *       -o rsx_vk_fallback_frag.spv.h rsx_vk_fallback.frag
 * then `expand -t 4` both outputs: glslang indents with tabs, the project
 * with spaces.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "rsx_vulkan_backend.h"
#include "rsx_vertex_fetch.h"
#include "rsx_texture_layout.h"
#include "rsx_vp_decompiler.h"
#include "rsx_fp_decompiler.h"
#include "rsx_shader_spirv.h"
#include "rsx_draw_engine.h"
#include "rsx_vertex_compact.h"
#include "ps3emu/ps3_ftrace.h"
#include "vulkan/rsx_vk_fallback_vert.spv.h"
#include "vulkan/rsx_vk_fallback_frag.spv.h"
#include "vulkan/rsx_vk_depth_rgba8_comp.spv.h"
#include "vulkan/rsx_vk_rgba8_rg16_comp.spv.h"
#include "vulkan/rsx_vk_vtx_decode_comp.spv.h"
#include "vulkan/rsx_vk_upscale_comp.spv.h"

/* SDL2 is already required off Windows (cellPad, cellAudio); here it only
 * opens the optional window. Same include style as those two. */
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#ifdef _WIN32
/* Windows: the Vulkan loader is vulkan-1.dll, resolved with LoadLibrary. */
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  define VK_DLOPEN(n)   ((void*)LoadLibraryA(n))
#  define VK_DLSYM(h, s) ((void*)GetProcAddress((HMODULE)(h), (s)))
#  define VK_DLCLOSE(h)  FreeLibrary((HMODULE)(h))
#  define VK_DLERROR()   "LoadLibrary failed"
#  define VK_LOADER_NAMES { "vulkan-1.dll" }
/* the queue/present locks below, as slim reader/writer locks */
typedef SRWLOCK pthread_mutex_t;
#  define pthread_mutex_init(m, a)  (InitializeSRWLock(m), 0)
#  define pthread_mutex_destroy(m)  ((void)(m))
#  define pthread_mutex_lock(m)     AcquireSRWLockExclusive(m)
#  define pthread_mutex_unlock(m)   ReleaseSRWLockExclusive(m)
#  define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#  define sched_yield() SwitchToThread()
#else
#  include <sched.h>
#  include <dlfcn.h>
#  include <pthread.h>
#  define VK_DLOPEN(n)   dlopen((n), RTLD_NOW | RTLD_LOCAL)
#  define VK_DLSYM(h, s) dlsym((h), (s))
#  define VK_DLCLOSE(h)  dlclose(h)
#  define VK_DLERROR()   dlerror()
#  ifdef __APPLE__
/* macOS: the Khronos loader over MoltenVK, as SDL loads it (SDL_Vulkan_*
 * must see the same library). Homebrew's prefix is not on dyld's default
 * search path, so it is named too. */
#    define VK_LOADER_NAMES { "libvulkan.1.dylib", "/opt/homebrew/lib/libvulkan.1.dylib", \
                              "/usr/local/lib/libvulkan.1.dylib", "libvulkan.dylib" }
#  else
#    define VK_LOADER_NAMES { "libvulkan.so.1", "libvulkan.so" }
#  endif
#endif
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>   /* _dyld_get_image_vmaddr_slide: ftrace caller addresses */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 * Run-time loaded entry points (Vulkan 1.0 core only)
 * -------------------------------------------------------------------------*/
#define VK_GLOBAL_FNS(X) X(vkCreateInstance)
#define VK_INSTANCE_FNS(X) \
    X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) \
    X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceFormatProperties) \
    X(vkGetPhysicalDeviceFeatures) \
    X(vkCreateDevice) X(vkGetDeviceProcAddr) X(vkEnumerateDeviceExtensionProperties)
#define VK_DEVICE_FNS(X) \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkDeviceWaitIdle) \
    X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements) X(vkBindImageMemory) \
    X(vkCreateImageView) X(vkDestroyImageView) \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory) X(vkUnmapMemory) \
    X(vkInvalidateMappedMemoryRanges) \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer) \
    X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences) X(vkQueueSubmit) \
    X(vkCreateRenderPass) X(vkDestroyRenderPass) X(vkCreateFramebuffer) X(vkDestroyFramebuffer) \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) \
    X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) \
    X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkResetDescriptorPool) \
    X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) \
    X(vkCreateGraphicsPipelines) X(vkCreateComputePipelines) X(vkDestroyPipeline) \
    X(vkCreateSampler) X(vkDestroySampler) \
    X(vkCmdPipelineBarrier) X(vkCmdClearColorImage) X(vkCmdClearDepthStencilImage) \
    X(vkCmdCopyImageToBuffer) X(vkCmdCopyBufferToImage) X(vkCmdCopyImage) \
    X(vkCmdBeginRenderPass) X(vkCmdEndRenderPass) X(vkCmdBindPipeline) \
    X(vkCmdBindVertexBuffers) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) \
    X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdDraw) \
    X(vkCmdSetStencilReference) X(vkCmdSetDepthBounds) X(vkCmdSetDepthBias) X(vkCmdSetBlendConstants) \
    X(vkCmdBindIndexBuffer) X(vkCmdDrawIndexed) \
    X(vkCreateSemaphore) X(vkDestroySemaphore) X(vkCmdBlitImage) X(vkCmdDispatch) X(vkCmdCopyBuffer)
/* Window-system entry points, loaded only when a window was requested, so a
 * headless run never depends on a driver that can present. */
#define VK_WSI_INSTANCE_FNS(X) \
    X(vkDestroySurfaceKHR) X(vkGetPhysicalDeviceSurfaceSupportKHR) \
    X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) X(vkGetPhysicalDeviceSurfaceFormatsKHR)
#define VK_WSI_DEVICE_FNS(X) \
    X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) X(vkGetSwapchainImagesKHR) \
    X(vkAcquireNextImageKHR) X(vkQueuePresentKHR)

#define VK_DECLARE(name) static PFN_##name p##name;
static PFN_vkGetInstanceProcAddr pvkGetInstanceProcAddr;
VK_GLOBAL_FNS(VK_DECLARE)
VK_INSTANCE_FNS(VK_DECLARE)
VK_DEVICE_FNS(VK_DECLARE)
VK_WSI_INSTANCE_FNS(VK_DECLARE)
VK_WSI_DEVICE_FNS(VK_DECLARE)

/* ---------------------------------------------------------------------------
 * Backend state
 * -------------------------------------------------------------------------*/
typedef struct vk_vertex {
    float pos[4];
    float col[4];
    float uv[2];
} vk_vertex;

typedef struct vk_buffer {
    VkBuffer       buf;
    VkDeviceMemory mem;
    void*          ptr;
    VkDeviceSize   size;
    int            coherent;
} vk_buffer;

typedef struct vk_image {
    VkImage        img;
    VkDeviceMemory mem;
    VkImageView    view;
} vk_image;

#define VK_SAMPLER_CACHE 32
#define VK_PIPELINE_SLOTS (2 * 2 * 8)   /* ztest x zwrite x compare op */

typedef struct vk_state {
    void*            lib;
    VkInstance       instance;
    VkPhysicalDevice phys;
    VkDevice         device;
    VkQueue          queue;
    u32              queue_family;
    VkPhysicalDeviceMemoryProperties memprops;

    u32              width, height;
    vk_image         color;           /* R8G8B8A8_UNORM                 */
    vk_image         depth;           /* D32_SFLOAT or fallback         */
    VkFormat         depth_format;
    VkImageAspectFlags depth_aspect;  /* DEPTH, plus STENCIL if it has one  */

    VkRenderPass     render_pass;
    VkFramebuffer    framebuffer;
    VkShaderModule   vs, fs;
    VkDescriptorSetLayout set_layout;
    VkDescriptorPool desc_pool;
    VkDescriptorSet  desc_set;
    VkPipelineLayout pipe_layout;
    VkPipeline       pipelines[VK_PIPELINE_SLOTS];
    VkSampler        sampler;

    vk_image         dummy;           /* 1x1 magenta: "nothing bound"    */
    vk_image         tex[RSX_MAX_TEXTURES];   /* decoded guest textures  */
    int              tex_ready[RSX_MAX_TEXTURES];
    /* Guest path only: each unit's sampler, built from its wrap, filter and
     * LOD registers (the fallback path keeps `sampler`, the null backend's
     * nearest + repeat). Cached by register value; VK_NULL_HANDLE = use
     * `sampler`. */
    VkSampler        tex_sampler[RSX_MAX_TEXTURES];
    struct { u64 key; VkSampler s; } samp_cache[VK_SAMPLER_CACHE];
    u32              samp_count;

    vk_buffer        vertices;        /* expanded triangle list          */
    vk_vertex*       cpu_verts;
    size_t           cpu_cap, cpu_count;

    vk_buffer        readback;
    VkCommandPool    pool;
    VkCommandBuffer  cmd;
    VkFence          fence;

    /* A present records and submits on its own pool, command buffer and
     * fences, so its second half can run on a host thread of its own while
     * the FIFO walker records the next frame (rsx_vulkan_backend.h,
     * present_begin/_finish). The queue is shared, and Vulkan wants every
     * vkQueueSubmit / vkQueuePresentKHR / vkDeviceWaitIdle on it externally
     * synchronised: queue_mu. present_mu keeps the present's own objects to
     * one present at a time, which matters when a flip in the command stream
     * presents on the walker's thread while a host thread finishes the last. */
    VkCommandPool    present_pool;
    VkCommandBuffer  present_cmd;
    VkFence          present_fence;   /* the copy out of the presented surface */
    VkFence          window_fence;    /* the blit into the swapchain image     */
    int              present_inflight;/* copy submitted, not yet finished      */
    int              present_split;   /* inside rsx_vulkan_backend_present_begin */
    int              present_order;   /* the walker's next command buffer waits */
    pthread_mutex_t  queue_mu;
    pthread_mutex_t  present_mu;

    /* Optional window (PS3RECOMP_VK_WINDOW=1). Rendering stays offscreen
     * either way -- readback and the tests are unchanged -- and a present
     * additionally blits the frame into the window's swapchain. */
    int              eng_active;      /* the register-file engine drives us */
    int              bc_ok;           /* textureCompressionBC enabled       */
    int              depth_bounds_ok; /* depthBounds enabled                */
    int              mirror_clamp_ok; /* VK_KHR_sampler_mirror_clamp_to_edge */
    u32              max_dim;         /* maxImageDimension2D of the device  */
    int              windowed;
    SDL_Window*      window;
    const char*      inst_exts[16];
    u32              inst_ext_count;
    VkSurfaceKHR     surface;
    VkSwapchainKHR   swapchain;
    VkExtent2D       sc_extent;
    u32              sc_count;
    VkImage          sc_images[8];
    VkSemaphore      sc_done[8];      /* per image: the present waits on it */
    VkSemaphore      sem_acquire;
    double           hold_seconds;

    const rsx_state* state;           /* live RSX state, for draws       */
    u32              clear_argb;
    u32              last_center;
    u32              presented;
    const char*      dump_path;
    u32              dump_every;       /* write every nth presented frame */
} vk_state;

static vk_state s_vk;

#define VK_LOG(...)  fprintf(stderr, "[RSX vulkan] " __VA_ARGS__)
#define VK_CHECK(call, what)                                               \
    do { VkResult r_ = (call);                                             \
         if (r_ != VK_SUCCESS) {                                           \
             VK_LOG("%s failed (VkResult %d)\n", (what), (int)r_);         \
             return -1; } } while (0)

/* ---------------------------------------------------------------------------
 * Loading
 * -------------------------------------------------------------------------*/
static int vk_load_library(void)
{
    static const char* names[] = VK_LOADER_NAMES;
    for (size_t i = 0; i < sizeof names / sizeof names[0] && !s_vk.lib; i++)
        s_vk.lib = VK_DLOPEN(names[i]);
    if (!s_vk.lib) {
        VK_LOG("no Vulkan loader found (%s)\n", VK_DLERROR());
        return -1;
    }
    pvkGetInstanceProcAddr =
        (PFN_vkGetInstanceProcAddr)VK_DLSYM(s_vk.lib, "vkGetInstanceProcAddr");
    if (!pvkGetInstanceProcAddr) { VK_LOG("libvulkan has no vkGetInstanceProcAddr\n"); return -1; }
#define LOAD_G(name) \
    if (!(p##name = (PFN_##name)pvkGetInstanceProcAddr(VK_NULL_HANDLE, #name))) \
        { VK_LOG("missing global function " #name "\n"); return -1; }
    VK_GLOBAL_FNS(LOAD_G)
    return 0;
}

static int vk_load_instance_functions(void)
{
#define LOAD_I(name) \
    if (!(p##name = (PFN_##name)pvkGetInstanceProcAddr(s_vk.instance, #name))) \
        { VK_LOG("missing instance function " #name "\n"); return -1; }
    VK_INSTANCE_FNS(LOAD_I)
    return 0;
}

static int vk_load_wsi_instance_functions(void)
{
    VK_WSI_INSTANCE_FNS(LOAD_I)
    return 0;
}

/* VK_SLOW_MS: device memory objects alive, counted through shims over
 * vkAllocateMemory/vkFreeMemory, so a slow call can be read against it. */
static volatile long s_vk_live_allocs;
static PFN_vkAllocateMemory s_vk_real_alloc;
static PFN_vkFreeMemory s_vk_real_free;
static VKAPI_ATTR VkResult VKAPI_CALL vk_count_alloc(VkDevice d, const VkMemoryAllocateInfo* ai,
                                                    const VkAllocationCallbacks* cb, VkDeviceMemory* m)
{
    const VkResult r = s_vk_real_alloc(d, ai, cb, m);
    if (r == VK_SUCCESS) __atomic_add_fetch(&s_vk_live_allocs, 1, __ATOMIC_RELAXED);
    return r;
}
static VKAPI_ATTR void VKAPI_CALL vk_count_free(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks* cb)
{
    if (m) __atomic_sub_fetch(&s_vk_live_allocs, 1, __ATOMIC_RELAXED);
    s_vk_real_free(d, m, cb);
}

static int vk_load_device_functions(void)
{
#define LOAD_D(name) \
    if (!(p##name = (PFN_##name)pvkGetDeviceProcAddr(s_vk.device, #name))) \
        { VK_LOG("missing device function " #name "\n"); return -1; }
    VK_DEVICE_FNS(LOAD_D)
    { const char* e = getenv("VK_SLOW_MS");
      if (e && atoi(e) > 0) {
          s_vk_real_alloc = pvkAllocateMemory; pvkAllocateMemory = vk_count_alloc;
          s_vk_real_free = pvkFreeMemory;      pvkFreeMemory = vk_count_free;
      } }
    return 0;
}

static int vk_load_wsi_device_functions(void)
{
    VK_WSI_DEVICE_FNS(LOAD_D)
    return 0;
}

/* ---------------------------------------------------------------------------
 * Device selection: PS3RECOMP_VK_DEVICE if set, else the first discrete GPU,
 * else integrated, else anything that is not a CPU rasteriser, else whatever
 * is there (llvmpipe/lavapipe on a GPU-less CI runner).
 * -------------------------------------------------------------------------*/
static int vk_device_rank(VkPhysicalDeviceType t)
{
    switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 4;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:            return 0;
    default:                                     return 1;
    }
}

static int vk_find_graphics_queue(VkPhysicalDevice pd, u32* family)
{
    u32 n = 0;
    pvkGetPhysicalDeviceQueueFamilyProperties(pd, &n, NULL);
    if (n == 0 || n > 64) return -1;
    VkQueueFamilyProperties props[64];
    pvkGetPhysicalDeviceQueueFamilyProperties(pd, &n, props);
    for (u32 i = 0; i < n; i++) {
        if (!(props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
        if (s_vk.windowed) {           /* the same queue must also present */
            VkBool32 ok = VK_FALSE;
            if (pvkGetPhysicalDeviceSurfaceSupportKHR(pd, i, s_vk.surface, &ok) != VK_SUCCESS || !ok)
                continue;
        }
        *family = i; return 0;
    }
    return -1;
}

static int vk_has_device_extension(VkPhysicalDevice pd, const char* name)
{
    u32 n = 0;
    if (pvkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL) != VK_SUCCESS || n == 0) return 0;
    VkExtensionProperties* e = (VkExtensionProperties*)malloc(n * sizeof *e);
    if (!e) return 0;
    int found = 0;
    if (pvkEnumerateDeviceExtensionProperties(pd, NULL, &n, e) == VK_SUCCESS)
        for (u32 i = 0; i < n && !found; i++) found = strcmp(e[i].extensionName, name) == 0;
    free(e);
    return found;
}

static int vk_pick_device(void)
{
    u32 n = 0;
    VK_CHECK(pvkEnumeratePhysicalDevices(s_vk.instance, &n, NULL), "vkEnumeratePhysicalDevices");
    if (n == 0) { VK_LOG("no Vulkan physical devices\n"); return -1; }
    if (n > 16) n = 16;
    VkPhysicalDevice devs[16];
    VK_CHECK(pvkEnumeratePhysicalDevices(s_vk.instance, &n, devs), "vkEnumeratePhysicalDevices");

    int best = -1, best_rank = -1;
    const char* forced = getenv("PS3RECOMP_VK_DEVICE");
    for (u32 i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        pvkGetPhysicalDeviceProperties(devs[i], &p);
        u32 fam;
        int usable = vk_find_graphics_queue(devs[i], &fam) == 0 &&
                     (!s_vk.windowed ||
                      vk_has_device_extension(devs[i], VK_KHR_SWAPCHAIN_EXTENSION_NAME));
        VK_LOG("device %u: %s (type %d, api %u.%u.%u)%s\n", i, p.deviceName,
               (int)p.deviceType, VK_API_VERSION_MAJOR(p.apiVersion),
               VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion),
               usable ? "" : (s_vk.windowed ? " -- cannot present to the window"
                                            : " -- no graphics queue"));
        if (!usable) continue;
        if (forced) {
            if ((u32)atoi(forced) == i) { best = (int)i; break; }
            continue;
        }
        int rank = vk_device_rank(p.deviceType);
        if (rank > best_rank) { best = (int)i; best_rank = rank; }
    }
    if (best < 0) {
        if (forced) VK_LOG("PS3RECOMP_VK_DEVICE=%s is not a usable device\n", forced);
        else        VK_LOG("no device with a graphics queue\n");
        return -1;
    }
    s_vk.phys = devs[best];
    vk_find_graphics_queue(s_vk.phys, &s_vk.queue_family);
    pvkGetPhysicalDeviceMemoryProperties(s_vk.phys, &s_vk.memprops);
    VkPhysicalDeviceProperties p;
    pvkGetPhysicalDeviceProperties(s_vk.phys, &p);
    /* Vulkan guarantees 4096; the engine's targets and textures may use what
     * the device really has (a title's 4608-wide target on RADV). */
    s_vk.max_dim = p.limits.maxImageDimension2D >= 4096u ? p.limits.maxImageDimension2D : 4096u;
    VK_LOG("using device %d: %s\n", best, p.deviceName);
    return 0;
}

/* ---------------------------------------------------------------------------
 * Memory, images, buffers
 * -------------------------------------------------------------------------*/
static int vk_find_memory_type(u32 type_bits, VkMemoryPropertyFlags want, u32* out)
{
    for (u32 i = 0; i < s_vk.memprops.memoryTypeCount; i++)
        if ((type_bits & (1u << i)) &&
            (s_vk.memprops.memoryTypes[i].propertyFlags & want) == want) {
            *out = i; return 0;
        }
    return -1;
}

static int vk_create_image_levels(vk_image* im, VkFormat fmt, u32 w, u32 h, u32 levels,
                                  VkImageUsageFlags usage, VkImageAspectFlags aspect)
{
    if (!levels) levels = 1;
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
        .format = fmt, .extent = { w, h, 1 }, .mipLevels = levels, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VK_CHECK(pvkCreateImage(s_vk.device, &ici, NULL, &im->img), "vkCreateImage");
    VkMemoryRequirements req;
    pvkGetImageMemoryRequirements(s_vk.device, im->img, &req);
    u32 type;
    if (vk_find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &type) &&
        vk_find_memory_type(req.memoryTypeBits, 0, &type)) {
        VK_LOG("no memory type for an image\n"); return -1;
    }
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = req.size, .memoryTypeIndex = type };
    VK_CHECK(pvkAllocateMemory(s_vk.device, &mai, NULL, &im->mem), "vkAllocateMemory(image)");
    VK_CHECK(pvkBindImageMemory(s_vk.device, im->img, im->mem, 0), "vkBindImageMemory");
    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im->img,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
        .subresourceRange = { aspect, 0, levels, 0, 1 },
    };
    VK_CHECK(pvkCreateImageView(s_vk.device, &vci, NULL, &im->view), "vkCreateImageView");
    return 0;
}

static int vk_create_image(vk_image* im, VkFormat fmt, u32 w, u32 h,
                           VkImageUsageFlags usage, VkImageAspectFlags aspect)
{
    return vk_create_image_levels(im, fmt, w, h, 1, usage, aspect);
}

static void vk_present_wait_gpu(void);

/* VK_DRAW_BATCH (vk_eng_draw): draws recorded into s_vk.cmd and not yet
 * submitted. Anything that records other work, destroys an object or reads
 * results flushes them first, so everything else keeps the invariant that
 * all earlier work has completed. */
static u32 s_eb_pending;
static int s_eb_async;           /* VK_ASYNC on: earlier batches may still be running */
static void vk_eng_flush(void);
static void vk_eng_idle(void);      /* flush, then wait for every batch in flight (VK_ASYNC) */
static void vk_eb_all_done(void);
static int vk_eb_active(void);       /* batching decided on and on */
static u32 vk_eb_max(void);
static void vk_eb_recorded(void);   /* one more command in the open batch */
static int vk_eb_upload(VkImage img, VkBufferImageCopy* region, const void* src, size_t tight,
                        u32 rows, u32 row_bytes);   /* -2: not batched, do it the old way */

/* VK_SLOW_MS=<ms>: log any driver call or wait below that takes longer
 * (lock, submit, fence, allocate, map, free, idle), with the thread. A
 * walker that stalls for seconds outside a fence wait shows up here. */
static int s_vk_slow_ms = -1;
static unsigned long long vk_slow_now(void)
{
    if (s_vk_slow_ms < 0) { const char* e = getenv("VK_SLOW_MS"); s_vk_slow_ms = e ? atoi(e) : 0; }
    if (s_vk_slow_ms <= 0) return 0;
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (unsigned long long)ts.tv_sec * 1000000ull + (unsigned long long)ts.tv_nsec / 1000ull;
}
static void vk_slow_note(const char* what, unsigned long long t0)
{
    if (!t0) return;
    const unsigned long long d = vk_slow_now() - t0;
    if (d < (unsigned long long)s_vk_slow_ms * 1000ull) return;
    static volatile long n;
    if (__atomic_add_fetch(&n, 1, __ATOMIC_RELAXED) > 3000) return;
#ifdef _WIN32
    fprintf(stderr, "[vk-slow] %s %.1f ms (thread %lu, %ld allocations)\n", what, d / 1000.0,
            (unsigned long)GetCurrentThreadId(), (long)s_vk_live_allocs);
#else
    fprintf(stderr, "[vk-slow] %s %.1f ms (%ld allocations)\n", what, d / 1000.0, (long)s_vk_live_allocs);
#endif
}

static void vk_destroy_image(vk_image* im)
{
    unsigned long long t = vk_slow_now();
    vk_eng_idle();
    vk_slow_note("destroy_image: idle", t);
    /* Never under a present's copy that may still read it. */
    t = vk_slow_now();
    if (im->img) vk_present_wait_gpu();
    vk_slow_note("destroy_image: present wait", t);
    t = vk_slow_now();
    if (im->view) pvkDestroyImageView(s_vk.device, im->view, NULL);
    if (im->img)  pvkDestroyImage(s_vk.device, im->img, NULL);
    if (im->mem)  pvkFreeMemory(s_vk.device, im->mem, NULL);
    vk_slow_note("destroy_image: free", t);
    memset(im, 0, sizeof *im);
}

static int vk_create_host_buffer(vk_buffer* b, VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                               .size = size, .usage = usage,
                               .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    VK_CHECK(pvkCreateBuffer(s_vk.device, &bci, NULL, &b->buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    pvkGetBufferMemoryRequirements(s_vk.device, b->buf, &req);
    u32 type;
    const VkMemoryPropertyFlags vis = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const VkMemoryPropertyFlags coh = vis | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    /* A buffer the GPU only copies into is one the CPU reads back, and the
     * first coherent type is usually write-combined and uncached (RADV's
     * GTT): there the readback memcpy was 82% of the FIFO thread's time in
     * Twisted Metal. Take a cached type for it when there is one; reads
     * invalidate, so a non-coherent one is fine. */
    const VkMemoryPropertyFlags cached = vis | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    const int readback = usage == VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (readback && vk_find_memory_type(req.memoryTypeBits, cached | coh, &type) == 0) b->coherent = 1;
    else if (readback && vk_find_memory_type(req.memoryTypeBits, cached, &type) == 0) b->coherent = 0;
    else if (vk_find_memory_type(req.memoryTypeBits, coh, &type) == 0) b->coherent = 1;
    else if (vk_find_memory_type(req.memoryTypeBits, vis, &type) == 0) b->coherent = 0;
    else { VK_LOG("no host-visible memory for a buffer\n"); return -1; }
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = req.size, .memoryTypeIndex = type };
    unsigned long long t = vk_slow_now();
    VK_CHECK(pvkAllocateMemory(s_vk.device, &mai, NULL, &b->mem), "vkAllocateMemory(buffer)");
    vk_slow_note("host buffer: allocate", t);
    VK_CHECK(pvkBindBufferMemory(s_vk.device, b->buf, b->mem, 0), "vkBindBufferMemory");
    t = vk_slow_now();
    VK_CHECK(pvkMapMemory(s_vk.device, b->mem, 0, VK_WHOLE_SIZE, 0, &b->ptr), "vkMapMemory");
    vk_slow_note("host buffer: map", t);
    b->size = size;
    return 0;
}

/* A buffer only the GPU touches (copies through it stay in VRAM). */
static int vk_create_device_buffer(vk_buffer* b, VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                               .size = size, .usage = usage,
                               .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    VK_CHECK(pvkCreateBuffer(s_vk.device, &bci, NULL, &b->buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    pvkGetBufferMemoryRequirements(s_vk.device, b->buf, &req);
    u32 type;
    if (vk_find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &type) &&
        vk_find_memory_type(req.memoryTypeBits, 0, &type)) { VK_LOG("no memory for a device buffer\n"); return -1; }
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = req.size, .memoryTypeIndex = type };
    VK_CHECK(pvkAllocateMemory(s_vk.device, &mai, NULL, &b->mem), "vkAllocateMemory(device buffer)");
    VK_CHECK(pvkBindBufferMemory(s_vk.device, b->buf, b->mem, 0), "vkBindBufferMemory");
    b->ptr = NULL; b->coherent = 0; b->size = size;
    return 0;
}

static void vk_destroy_buffer(vk_buffer* b)
{
    unsigned long long t = vk_slow_now();
    vk_eng_idle();
    vk_slow_note("destroy_buffer: idle", t);
    t = vk_slow_now();
    if (b->ptr) pvkUnmapMemory(s_vk.device, b->mem);
    if (b->buf) pvkDestroyBuffer(s_vk.device, b->buf, NULL);
    if (b->mem) pvkFreeMemory(s_vk.device, b->mem, NULL);
    vk_slow_note("destroy_buffer: free", t);
    memset(b, 0, sizeof *b);
}

/* Host writes to non-coherent memory need a flush; there is no
 * vkFlushMappedMemoryRanges loaded, so writes only ever go to coherent
 * buffers (every desktop/L4T driver exposes one). Reads invalidate. */
static void vk_invalidate(vk_buffer* b)
{
    if (b->coherent) return;
    VkMappedMemoryRange r = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                              .memory = b->mem, .offset = 0, .size = VK_WHOLE_SIZE };
    pvkInvalidateMappedMemoryRanges(s_vk.device, 1, &r);
}

/* ---------------------------------------------------------------------------
 * One-shot command submission
 *
 * The queue is shared with the present (see present_mu in vk_state), so every
 * operation on it goes through these.
 * -------------------------------------------------------------------------*/
/* Every queue submission's sequence number (one queue, executed in order). */
static volatile uint64_t s_vk_sub_seq;
static VkResult vk_queue_submit(const VkSubmitInfo* si, VkFence fence)
{
    unsigned long long t = vk_slow_now();
    pthread_mutex_lock(&s_vk.queue_mu);
    vk_slow_note("submit: queue lock", t);
    t = vk_slow_now();
    const VkResult r = pvkQueueSubmit(s_vk.queue, 1, si, fence);
    vk_slow_note("submit: vkQueueSubmit", t);
    if (r == VK_SUCCESS) s_vk_sub_seq++;
    pthread_mutex_unlock(&s_vk.queue_mu);
    return r;
}

/* Deferred texture destruction (VK_TEX_DEFER=0: destroy at once, after waiting
 * for the GPU, as before). A released texture may still be sampled by the
 * batches in flight and by the open one, so destroying it meant vk_eng_idle()
 * -- a wait for every batch -- per release: ~40 a second for GoW:A's grading
 * LUT and hundreds while a scene streams textures into a full cache. Instead it
 * waits here, tagged with the first submission that can follow its last use,
 * and goes once a batch fence at or past that has been seen to signal. Walker
 * thread only (the engine's own submissions; a present never samples one). */
#define VK_GRAVE_N 4096u
static struct { VkImage img; VkImageView view; VkDeviceMemory mem; uint64_t seq; } s_grave[VK_GRAVE_N];
static u32 s_grave_n;
static int vk_tex_defer(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("VK_TEX_DEFER"); on = !(e && *e == '0'); }
    return on;
}
static void vk_grave_destroy(u32 i)
{
    if (s_grave[i].view) pvkDestroyImageView(s_vk.device, s_grave[i].view, NULL);
    if (s_grave[i].img)  pvkDestroyImage(s_vk.device, s_grave[i].img, NULL);
    if (s_grave[i].mem)  pvkFreeMemory(s_vk.device, s_grave[i].mem, NULL);
}
/* Every submission up to `done` has completed: destroy what waited for it. */
static void vk_grave_done(uint64_t done)
{
    u32 k = 0; int waited = 0;
    for (u32 i = 0; i < s_grave_n; i++) {
        if (s_grave[i].seq <= done) {
            /* As vk_destroy_image: never under a present's copy that may read it. */
            if (!waited) { vk_present_wait_gpu(); waited = 1; }
            vk_grave_destroy(i);
        } else s_grave[k++] = s_grave[i];
    }
    s_grave_n = k;
}
/* 1 = queued, 0 = the caller destroys it the old way. */
static int vk_grave_push(VkImage img, VkImageView view, VkDeviceMemory mem)
{
    if (!vk_tex_defer() || s_grave_n >= VK_GRAVE_N) return 0;
    s_grave[s_grave_n].img = img; s_grave[s_grave_n].view = view; s_grave[s_grave_n].mem = mem;
    s_grave[s_grave_n].seq = s_vk_sub_seq + 1u;   /* the next submission carries the open batch */
    s_grave_n++;
    return 1;
}

static VkResult vk_queue_present(const VkPresentInfoKHR* pi)
{
    pthread_mutex_lock(&s_vk.queue_mu);
    const VkResult r = pvkQueuePresentKHR(s_vk.queue, pi);
    pthread_mutex_unlock(&s_vk.queue_mu);
    return r;
}

static void vk_device_idle(void)
{
    if (!s_vk.device || !pvkDeviceWaitIdle) return;
    pthread_mutex_lock(&s_vk.queue_mu);
    const unsigned long long t = vk_slow_now();
    pvkDeviceWaitIdle(s_vk.device);
    vk_slow_note("device idle (queue lock held)", t);
    pthread_mutex_unlock(&s_vk.queue_mu);
}

/* VK_PASS_MERGE=1 (vk_eng_draw): a batched draw leaves its render pass open,
 * and the next one records into it when it has the same render pass,
 * framebuffer and size and samples none of the pass's targets. Draws used to
 * get a pass, a full memory barrier and an image barrier per target each,
 * ~2000 times a frame. Inside one pass, rasterization order keeps blending
 * and depth right; a draw that samples a target of the open pass still gets
 * a new pass behind the full barrier. Every other command recorded into
 * s_vk.cmd ends the pass first: the barrier helpers below, vk_begin, the
 * submits and vk_eng_flush call vk_pass_end. */
static struct {
    int open;
    VkRenderPass rp;
    VkFramebuffer fb;
    u32 w, h, n, rt[4], depth;
} s_pass;
static int s_pass_merge = -1;
static unsigned long long s_pass_n[3];      /* passes begun, draws merged, breaks for a sampled target */
static int vk_pass_merge(void)
{
    if (s_pass_merge < 0) { const char* e = getenv("VK_PASS_MERGE"); s_pass_merge = e ? atoi(e) : 0; }
    return s_pass_merge;
}
static void vk_pass_end(void)
{
    if (!s_pass.open) return;
    s_pass.open = 0;
    pvkCmdEndRenderPass(s_vk.cmd);
}

/* A full execution and memory dependency on everything submitted before
 * this command buffer, the present's copy included. */
static void vk_cmd_after_everything(VkCommandBuffer cb)
{
    if (cb == s_vk.cmd) vk_pass_end();
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                           .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
    pvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

/* Bumped whenever a command buffer starts or another path sets graphics
 * state in it: vk_eng_draw's record of what is already bound (VK_STATE_SKIP)
 * is trusted only within one epoch. */
static u32 s_rec_epoch;

/* VK_UB_DYNAMIC=1 (read once, before any layout exists): the two uniform
 * bindings are UNIFORM_BUFFER_DYNAMIC, so a draw passes its uniform slices as
 * dynamic offsets at bind time and vk_eng_draw can bind the previous draw's
 * descriptor set again when its views and samplers are the same. */
static int vk_ub_dynamic(void)
{
    static int s = -1;
    if (s < 0) { const char* e = getenv("VK_UB_DYNAMIC"); s = e ? atoi(e) : 0; }
    return s;
}
#define VK_UB_TYPE (vk_ub_dynamic() ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
/* VK_CB_REUSE=1 (with VK_UB_DYNAMIC): a draw bound with no constants since the
 * previous one reuses that draw's uniform slices (see vk_eng_draw). */
static int vk_cb_reuse(void)
{
    static int s = -1;
    if (s < 0) { const char* e = getenv("VK_CB_REUSE"); s = e && e[0] == '1'; }
    return s;
}

static int vk_begin(void)
{
    s_rec_epoch++;
    if (s_eb_pending) vk_eng_flush();
    vk_pass_end();
    VK_CHECK(pvkResetCommandBuffer(s_vk.cmd, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VK_CHECK(pvkBeginCommandBuffer(s_vk.cmd, &bi), "vkBeginCommandBuffer");
    /* A present's copy may still be reading the surface this frame is about
     * to draw over: order everything here after it. Set only by a present,
     * which never runs alongside the walker's recording. */
    if (s_vk.present_order || s_eb_async) { vk_cmd_after_everything(s_vk.cmd); s_vk.present_order = 0; }
    return 0;
}

/* Nonzero while the backend waits for the GPU (read by a port watchdog). */
volatile int g_vk_fence_waiting = 0;

/* PS3_FTRACE: each synchronous submit is a GPU_WAIT span (a = 3, b = the
 * caller's return address as a module RVA), so a timeline shows how much of
 * the walker's time is spent waiting on the GPU and from which call site. */
static __attribute__((noinline)) int vk_submit_and_wait(void)
{
    const uint64_t ft_t0 = PS3_FT_ON() ? ps3_ft_now() : 0;
#ifdef _WIN32
    const uint32_t ft_ra = ft_t0 ? (uint32_t)((uintptr_t)__builtin_return_address(0) -
                                              (uintptr_t)GetModuleHandleA(NULL)) : 0;
#elif defined(__APPLE__)
    /* The static address less the 0x100000000 base, as `atos -o <exe> -l 0x100000000
     * 0x1<b>` takes it: ASLR's slide removed. */
    const uint32_t ft_ra = ft_t0 ? (uint32_t)((uintptr_t)__builtin_return_address(0) -
                                              (uintptr_t)_dyld_get_image_vmaddr_slide(0) - 0x100000000ull) : 0;
#else
    const uint32_t ft_ra = (uint32_t)(uintptr_t)__builtin_return_address(0);
#endif
    vk_pass_end();
    VK_CHECK(pvkEndCommandBuffer(s_vk.cmd), "vkEndCommandBuffer");
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1, .pCommandBuffers = &s_vk.cmd };
    VK_CHECK(pvkResetFences(s_vk.device, 1, &s_vk.fence), "vkResetFences");
    VK_CHECK(vk_queue_submit(&si, s_vk.fence), "vkQueueSubmit");
    const uint64_t my_seq = s_vk_sub_seq;
    g_vk_fence_waiting = 1;
    const unsigned long long sw = vk_slow_now();
    VkResult wr = pvkWaitForFences(s_vk.device, 1, &s_vk.fence, VK_TRUE, UINT64_MAX);
    if (wr == VK_SUCCESS && s_grave_n) vk_grave_done(my_seq);
    vk_slow_note("submit_and_wait: fence", sw);
    g_vk_fence_waiting = 0;
    if (ft_t0) {
        const uint64_t ft_t1 = ps3_ft_now();
        if (ft_t1 - ft_t0 >= ps3_ft_min_ns()) ps3_ft_span(PS3_FT_GPU_WAIT, 0, 3, ft_ra, ft_t0, ft_t1);
    }
    VK_CHECK(wr, "vkWaitForFences");
    vk_eb_all_done();              /* the fence covers everything queued before */
    return 0;
}

/* Order this command buffer's use of an image after everything submitted
 * before it; from UNDEFINED it is also the one-time move to GENERAL. */
static void vk_barrier_image_on(VkCommandBuffer cb, VkImage img, VkImageAspectFlags aspect,
                                VkImageLayout old_layout)
{
    if (cb == s_vk.cmd) vk_pass_end();
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .oldLayout = old_layout, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img,
        .subresourceRange = { aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS },
    };
    pvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
}

static void vk_barrier_image(VkImage img, VkImageAspectFlags aspect, VkImageLayout old_layout)
{
    vk_barrier_image_on(s_vk.cmd, img, aspect, old_layout);
}

/* ---------------------------------------------------------------------------
 * Texture unit 0
 * -------------------------------------------------------------------------*/
static void vk_write_texture_descriptor(void)
{
    VkDescriptorImageInfo ii = { .sampler = s_vk.sampler,
                                 .imageView = s_vk.tex_ready[0] ? s_vk.tex[0].view
                                                                : s_vk.dummy.view,
                                 .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet w = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                               .dstSet = s_vk.desc_set, .dstBinding = 0,
                               .descriptorCount = 1,
                               .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                               .pImageInfo = &ii };
    pvkUpdateDescriptorSets(s_vk.device, 1, &w, 0, NULL);
}

/* Replace *dst with an RGBA8 image of `nlv` levels: level m is lw[m] x lh[m]
 * texels, tightly packed, the levels one after another in `rgba`. */
static int vk_upload_texture_levels(vk_image* dst, const u8* rgba,
                                    const u32* lw, const u32* lh, u32 nlv)
{
    VkBufferImageCopy regions[RSX_MAX_TEXTURE_LEVELS];
    if (!nlv || nlv > RSX_MAX_TEXTURE_LEVELS) return -1;
    VkDeviceSize total = 0;
    for (u32 m = 0; m < nlv; m++) {
        regions[m] = (VkBufferImageCopy){
            .bufferOffset = total,
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1 },
            .imageExtent = { lw[m], lh[m], 1 },
        };
        total += (VkDeviceSize)lw[m] * lh[m] * 4u;
    }
    vk_buffer staging = {0};
    if (vk_create_host_buffer(&staging, total, VK_BUFFER_USAGE_TRANSFER_SRC_BIT) ||
        !staging.coherent) {
        vk_destroy_buffer(&staging); return -1;
    }
    memcpy(staging.ptr, rgba, (size_t)total);

    vk_image fresh = {0};
    if (vk_create_image_levels(&fresh, VK_FORMAT_R8G8B8A8_UNORM, lw[0], lh[0], nlv,
                               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT)) {
        vk_destroy_image(&fresh); vk_destroy_buffer(&staging); return -1;
    }
    int rc = vk_begin();
    if (!rc) {
        vk_barrier_image(fresh.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
        pvkCmdCopyBufferToImage(s_vk.cmd, staging.buf, fresh.img, VK_IMAGE_LAYOUT_GENERAL,
                                nlv, regions);
        rc = vk_submit_and_wait();
    }
    vk_destroy_buffer(&staging);
    if (rc) { vk_destroy_image(&fresh); return -1; }

    vk_destroy_image(dst);          /* idle: every submission is waited on */
    *dst = fresh;
    return 0;
}

static int vk_upload_texture(vk_image* dst, const u8* rgba, u32 w, u32 h)
{
    return vk_upload_texture_levels(dst, rgba, &w, &h, 1);
}

/* NV4097_SET_TEXTURE_ADDRESS wrap field, one per axis -- the table the Metal
 * backend and the live draw engine use: 1 WRAP, 2 MIRROR, 3 CLAMP_TO_EDGE,
 * 4 BORDER, 5 CLAMP, 6..8 MIRROR_ONCE. Vulkan 1.0 has MIRROR_CLAMP_TO_EDGE
 * only behind an extension, so the MIRROR_ONCE family clamps to edge here --
 * exact for texcoords in [0,1], the mirrored half is what is lost. Where the
 * device has VK_KHR_sampler_mirror_clamp_to_edge (MoltenVK: Metal's
 * mirrorClampToEdge) the family mirrors once, as RSX does: a texture laid on
 * [-1,1] with MIRROR_ONCE otherwise smears its edge texels over the negative
 * half (GoW:A's illusion arena after Megaera). RSX_VK_MIRROR_ONCE=0 clamps. */
static VkSamplerAddressMode vk_gcm_wrap(u32 w)
{
    switch (w & 0xFu) {
    case 1:  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case 2:  return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case 4:  return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case 6: case 7: case 8:
        if (s_vk.mirror_clamp_ok) return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;      /* 3, 5..8, unset */
    }
}

/* A unit's sampler from its registers, decoded as the Metal backend's
 * samp_slot_for does: SET_TEXTURE_FILTER min at [18:16] (1 NEAREST,
 * 2 LINEAR, 3..6 the four nearest/linear x nearest/linear-mip pairs), mag at
 * [26:24]; SET_TEXTURE_CONTROL0 max LOD at [18:7], min LOD at [30:19], 4.8
 * fixed point. The LOD bias (FILTER [12:0]) is not applied, as in Metal.
 * VK_NULL_HANDLE on failure: the caller falls back to `sampler`. */
static VkSampler vk_unit_sampler(const rsx_texture_state* t)
{
    const u32 minf = (t->filter >> 16) & 7u, magf = (t->filter >> 24) & 7u;
    const u32 lod  = (t->control0 >> 7) & 0xFFFFFFu;       /* max then min LOD */
    const u64 key  = (u64)minf | ((u64)magf << 3)
                   | ((u64)(t->address & 0x000F0F0Fu) << 6) | ((u64)lod << 26);
    for (u32 i = 0; i < s_vk.samp_count; i++)
        if (s_vk.samp_cache[i].key == key) return s_vk.samp_cache[i].s;
    if (s_vk.samp_count >= VK_SAMPLER_CACHE) return VK_NULL_HANDLE;

    const int mip_present = (minf >= 3);
    const int mip_linear  = (minf == 5 || minf == 6);
    /* A min filter with no mip term samples level 0 only: pin the range. */
    const float min_lod = (float)((t->control0 >> 19) & 0xFFFu) / 256.0f;
    float max_lod = mip_present ? (float)((t->control0 >> 7) & 0xFFFu) / 256.0f : 0.0f;
    if (max_lod < min_lod) max_lod = min_lod;
    VkSamplerCreateInfo sci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = (magf == 2) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .minFilter = (minf == 2 || minf == 4 || minf == 6) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .mipmapMode = mip_linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = vk_gcm_wrap(t->address),
        .addressModeV = vk_gcm_wrap(t->address >> 8),
        .addressModeW = vk_gcm_wrap(t->address >> 16),
        .minLod = min_lod, .maxLod = max_lod,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
    };
    VkSampler smp = VK_NULL_HANDLE;
    if (pvkCreateSampler(s_vk.device, &sci, NULL, &smp) != VK_SUCCESS) return VK_NULL_HANDLE;
    s_vk.samp_cache[s_vk.samp_count].key = key;
    s_vk.samp_cache[s_vk.samp_count].s   = smp;
    s_vk.samp_count++;
    return smp;
}

/* One level of a guest texture into tightly packed RGBA8, with the null
 * backend's decode (rsx_null_backend.c, nullsw_bind_texture): formats that
 * decode to fewer than four bytes are splayed, alpha forced opaque. */
static int vk_decode_level_rgba8(u8* dst, const u8* src, u32 w, u32 h,
                                 const rsx_tex_layout* tl)
{
    const u32 pitch = w * 4u;
    if (tl->fmt == RSX_TEXFMT_R8G8B8A8) {
        rsx_texture_decode(dst, pitch, src, w, h, tl, rsx_texture_argb_is_rgba());
        return 0;
    }
    const u32 srcp = tl->row_bytes;
    u8* tmp = (u8*)malloc((size_t)srcp * h);
    if (!tmp) return -1;
    rsx_texture_decode(tmp, srcp, src, w, h, tl, 0);
    for (u32 y = 0; y < h; y++)
        for (u32 x = 0; x < w; x++) {
            const u8* sp = tmp + (size_t)y * srcp + (size_t)x * tl->bytes_per_texel;
            u8* dp = dst + (size_t)y * pitch + (size_t)x * 4u;
            dp[0] = sp[0];
            dp[1] = tl->bytes_per_texel > 1 ? sp[1] : sp[0];
            dp[2] = tl->bytes_per_texel > 2 ? sp[2] : sp[0];
            dp[3] = 255;
        }
    free(tmp);
    return 0;
}

/* Unit `unit` has nothing usable bound. The fallback path's descriptor only
 * ever shows unit 0, so only that one needs rewriting. */
static void vk_tex_off(u32 unit)
{
    s_vk.tex_ready[unit] = 0;
    s_vk.tex_sampler[unit] = VK_NULL_HANDLE;
    if (unit == 0) vk_write_texture_descriptor();
}

/* Same decode as the headless null backend (rsx_null_backend.c,
 * nullsw_bind_texture), so both backends sample identical texels. */
static void vk_cb_bind_texture(void* ud, u32 unit, const rsx_texture_state* t)
{
    (void)ud;
    if (unit >= RSX_MAX_TEXTURES || !t || !s_vk.device) return;
    if (!(t->control0 & 0x80000000u) || !t->offset) { vk_tex_off(unit); return; }

    /* VK_TEX_OFF=1: never decode/upload guest textures (diagnosis). */
    { static int off = -1; if (off < 0) { const char* e = getenv("VK_TEX_OFF"); off = e && *e != '0'; }
      if (off) { vk_tex_off(unit); return; } }
    u32 w = (t->image_rect >> 16) & 0xFFFFu;
    u32 h =  t->image_rect        & 0xFFFFu;
    if (!w || !h || w > 4096u || h > 4096u) { vk_tex_off(unit); return; }

    u32 ea = cellGcmResolveLocated((t->format & 3u) == 1u, t->offset);
    if (!vm_base || ea == 0xFFFFFFFFu) { vk_tex_off(unit); return; }
    const u32 fmt = (t->format >> 8) & 0xFFu;

    /* The whole mip chain, as the Metal backend reads it: SET_TEXTURE_FORMAT's
     * level count above bit 15 and SET_TEXTURE_CONTROL3's row pitch, laid out
     * by the shared rsx_texture_mip_chain() (which clamps the count to what
     * the dimensions allow). Cube maps are not handled yet: face 0 only. */
    const u32 levels    = (t->format >> 16) & 0xFFFFu;
    const u32 row_pitch = t->control3 & 0xFFFFFu;
    rsx_tex_level lv[RSX_MAX_TEXTURE_LEVELS];
    const u32 nlv = rsx_texture_mip_chain(fmt, w, h, levels, row_pitch, lv);
    if (!nlv || lv[0].tl.compressed) { vk_tex_off(unit); return; }   /* no BC path yet */

    size_t total = 0, off[RSX_MAX_TEXTURE_LEVELS];
    u32 lw[RSX_MAX_TEXTURE_LEVELS], lh[RSX_MAX_TEXTURE_LEVELS];
    for (u32 m = 0; m < nlv; m++) {
        off[m] = total;
        lw[m] = lv[m].w; lh[m] = lv[m].h;
        total += (size_t)lv[m].w * lv[m].h * 4u;
    }
    u8* buf = (u8*)malloc(total);
    if (!buf) { vk_tex_off(unit); return; }
    for (u32 m = 0; m < nlv; m++)
        if (vk_decode_level_rgba8(buf + off[m], vm_base + ea + lv[m].offset,
                                  lv[m].w, lv[m].h, &lv[m].tl)) {
            free(buf); vk_tex_off(unit); return;
        }
    s_vk.tex_ready[unit] = vk_upload_texture_levels(&s_vk.tex[unit], buf, lw, lh, nlv) == 0;
    free(buf);
    s_vk.tex_sampler[unit] = s_vk.tex_ready[unit] ? vk_unit_sampler(t) : VK_NULL_HANDLE;
    if (unit == 0) vk_write_texture_descriptor();
}

/* ---------------------------------------------------------------------------
 * Render pass, framebuffer, pipelines
 * -------------------------------------------------------------------------*/
static int vk_pick_depth_format(void)
{
    /* With stencil first -- titles use it for shadows, reflections and UI
     * masks -- and D32 first within each group: a D32 depth aspect copies out
     * as 32-bit floats, which is what depth snapshots are built from. */
    static const VkFormat candidates[] = {
        VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D32_SFLOAT, VK_FORMAT_X8_D24_UNORM_PACK32, VK_FORMAT_D16_UNORM
    };
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    for (size_t i = 0; i < sizeof candidates / sizeof candidates[0]; i++) {
        VkFormatProperties fp;
        pvkGetPhysicalDeviceFormatProperties(s_vk.phys, candidates[i], &fp);
        if ((fp.optimalTilingFeatures & need) == need) {
            s_vk.depth_format = candidates[i]; goto chosen;
        }
    }
    /* TRANSFER_DST is a 1.1 format feature bit; 1.0 drivers may not report
     * it although clears work. Retry on the attachment bit alone. */
    for (size_t i = 0; i < sizeof candidates / sizeof candidates[0]; i++) {
        VkFormatProperties fp;
        pvkGetPhysicalDeviceFormatProperties(s_vk.phys, candidates[i], &fp);
        if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            s_vk.depth_format = candidates[i]; goto chosen;
        }
    }
    VK_LOG("no usable depth format\n");
    return -1;
chosen:
    s_vk.depth_aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (s_vk.depth_format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
        s_vk.depth_format == VK_FORMAT_D24_UNORM_S8_UINT)
        s_vk.depth_aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    return 0;
}

static int vk_create_render_pass(void)
{
    VkAttachmentDescription att[2] = {
        { .format = VK_FORMAT_R8G8B8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
          .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
          .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
          .initialLayout = VK_IMAGE_LAYOUT_GENERAL, .finalLayout = VK_IMAGE_LAYOUT_GENERAL },
        { .format = s_vk.depth_format, .samples = VK_SAMPLE_COUNT_1_BIT,
          .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD,     /* ignored without stencil */
          .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
          .initialLayout = VK_IMAGE_LAYOUT_GENERAL, .finalLayout = VK_IMAGE_LAYOUT_GENERAL },
    };
    VkAttachmentReference cref = { 0, VK_IMAGE_LAYOUT_GENERAL };
    VkAttachmentReference dref = { 1, VK_IMAGE_LAYOUT_GENERAL };
    VkSubpassDescription sub = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 .colorAttachmentCount = 1, .pColorAttachments = &cref,
                                 .pDepthStencilAttachment = &dref };
    VkRenderPassCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                                   .attachmentCount = 2, .pAttachments = att,
                                   .subpassCount = 1, .pSubpasses = &sub };
    VK_CHECK(pvkCreateRenderPass(s_vk.device, &rci, NULL, &s_vk.render_pass), "vkCreateRenderPass");

    VkImageView views[2] = { s_vk.color.view, s_vk.depth.view };
    VkFramebufferCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                    .renderPass = s_vk.render_pass, .attachmentCount = 2,
                                    .pAttachments = views, .width = s_vk.width,
                                    .height = s_vk.height, .layers = 1 };
    VK_CHECK(pvkCreateFramebuffer(s_vk.device, &fci, NULL, &s_vk.framebuffer), "vkCreateFramebuffer");
    return 0;
}

static int vk_create_shader(const uint32_t* code, size_t bytes, VkShaderModule* out)
{
    VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                    .codeSize = bytes, .pCode = code };
    VK_CHECK(pvkCreateShaderModule(s_vk.device, &ci, NULL, out), "vkCreateShaderModule");
    return 0;
}

static int vk_create_pipeline_objects(void)
{
    if (vk_create_shader(rsx_vk_fallback_vert_spv, sizeof rsx_vk_fallback_vert_spv, &s_vk.vs) ||
        vk_create_shader(rsx_vk_fallback_frag_spv, sizeof rsx_vk_fallback_frag_spv, &s_vk.fs))
        return -1;

    VkDescriptorSetLayoutBinding bind = {
        .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
    VkDescriptorSetLayoutCreateInfo lci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &bind };
    VK_CHECK(pvkCreateDescriptorSetLayout(s_vk.device, &lci, NULL, &s_vk.set_layout),
             "vkCreateDescriptorSetLayout");

    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
    VK_CHECK(pvkCreateDescriptorPool(s_vk.device, &pci, NULL, &s_vk.desc_pool), "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = s_vk.desc_pool,
                                        .descriptorSetCount = 1, .pSetLayouts = &s_vk.set_layout };
    VK_CHECK(pvkAllocateDescriptorSets(s_vk.device, &dai, &s_vk.desc_set), "vkAllocateDescriptorSets");

    VkPushConstantRange pcr = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int32_t) };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1, .pSetLayouts = &s_vk.set_layout,
                                        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    VK_CHECK(pvkCreatePipelineLayout(s_vk.device, &plci, NULL, &s_vk.pipe_layout),
             "vkCreatePipelineLayout");

    /* Nearest + repeat: the null backend's point sample with wrapping. */
    VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
                                .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                .maxLod = 0.0f };
    VK_CHECK(pvkCreateSampler(s_vk.device, &sci, NULL, &s_vk.sampler), "vkCreateSampler");

    /* A 1x1 magenta dummy keeps the descriptor valid before any bind --
     * the same "not bound" colour the null backend's sampler returns. */
    static const u8 magenta[4] = { 0xFF, 0x00, 0xFF, 0xFF };
    if (vk_upload_texture(&s_vk.dummy, magenta, 1, 1)) return -1;
    vk_write_texture_descriptor();
    return 0;
}

/* NV4097 depth function is the GL enum 0x200 NEVER .. 0x207 ALWAYS, which is
 * VkCompareOp 0..7 in the same order. Anything else passes, as in the null
 * and D3D12 backends (including the 1 rsx_state_init seeds). */
static u32 vk_compare_slot(u32 gl_func)
{
    return (gl_func >= 0x200u && gl_func <= 0x207u) ? gl_func - 0x200u
                                                    : (u32)VK_COMPARE_OP_ALWAYS;
}

static VkPipeline vk_get_pipeline(int ztest, int zwrite, u32 cmp)
{
    u32 slot = ((u32)ztest * 2u + (u32)zwrite) * 8u + cmp;
    if (s_vk.pipelines[slot]) return s_vk.pipelines[slot];

    VkPipelineShaderStageCreateInfo stages[2] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = s_vk.vs, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = s_vk.fs, .pName = "main" },
    };
    VkVertexInputBindingDescription vb = { 0, sizeof(vk_vertex), VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription va[3] = {
        { 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(vk_vertex, pos) },
        { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(vk_vertex, col) },
        { 2, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(vk_vertex, uv)  },
    };
    VkPipelineVertexInputStateCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vb,
        .vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = va };
    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkPipelineViewportStateCreateInfo vp = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1 };
    /* No culling: the null backend fills both windings. */
    VkPipelineRasterizationStateCreateInfo rs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkPipelineDepthStencilStateCreateInfo ds = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = ztest ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = zwrite ? VK_TRUE : VK_FALSE,
        .depthCompareOp = (VkCompareOp)cmp };
    VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xF };
    VkPipelineColorBlendStateCreateInfo cb = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &cba };
    VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2, .pDynamicStates = dyn };
    VkGraphicsPipelineCreateInfo gci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = stages, .pVertexInputState = &vi,
        .pInputAssemblyState = &ia, .pViewportState = &vp, .pRasterizationState = &rs,
        .pMultisampleState = &ms, .pDepthStencilState = &ds, .pColorBlendState = &cb,
        .pDynamicState = &dy, .layout = s_vk.pipe_layout, .renderPass = s_vk.render_pass,
        .subpass = 0 };
    VkPipeline p = VK_NULL_HANDLE;
    VkResult r = pvkCreateGraphicsPipelines(s_vk.device, VK_NULL_HANDLE, 1, &gci, NULL, &p);
    if (r != VK_SUCCESS) { VK_LOG("vkCreateGraphicsPipelines failed (%d)\n", (int)r); return VK_NULL_HANDLE; }
    s_vk.pipelines[slot] = p;
    return p;
}

/* ---------------------------------------------------------------------------
 * Draw path
 * -------------------------------------------------------------------------*/
static void vk_cb_track_state(void* ud, const rsx_state* state)
{
    (void)ud; s_vk.state = state;
}

static int vk_push_vertex(const float pos[4], const float col[4], const float uv[4])
{
    if (s_vk.cpu_count == s_vk.cpu_cap) {
        size_t cap = s_vk.cpu_cap ? s_vk.cpu_cap * 2 : 1024;
        vk_vertex* n = (vk_vertex*)realloc(s_vk.cpu_verts, cap * sizeof *n);
        if (!n) return -1;
        s_vk.cpu_verts = n; s_vk.cpu_cap = cap;
    }
    vk_vertex* v = &s_vk.cpu_verts[s_vk.cpu_count++];
    memcpy(v->pos, pos, sizeof v->pos);
    memcpy(v->col, col, sizeof v->col);
    v->uv[0] = uv[0]; v->uv[1] = uv[1];
    return 0;
}

/* One triangle, colour from its first vertex (the null backend's flat shade). */
static void vk_emit_tri(const rsx_state* st, u32 i0, u32 i1, u32 i2)
{
    float p[3][4], uv[3][4], col[4];
    rsx_fetch_attrib(st, 0, i0, p[0]);
    rsx_fetch_attrib(st, 0, i1, p[1]);
    rsx_fetch_attrib(st, 0, i2, p[2]);
    rsx_fetch_attrib(st, 3, i0, col);
    rsx_fetch_attrib(st, 8, i0, uv[0]);
    rsx_fetch_attrib(st, 8, i1, uv[1]);
    rsx_fetch_attrib(st, 8, i2, uv[2]);
    for (int k = 0; k < 3; k++) vk_push_vertex(p[k], col, uv[k]);
}

/* Called once per expanded triangle: the fixed-function path records the
 * null backend's flat-shaded vertex, the guest path all sixteen attributes. */
typedef void (*vk_tri_fn)(const rsx_state* st, u32 i0, u32 i1, u32 i2);

/* Primitive expansion, identical to the null backend's draw_prim(). */
static void vk_expand(const rsx_state* st, u32 prim, u32 first, u32 count, vk_tri_fn emit)
{
    switch (prim) {
    case RSX_PRIMITIVE_TRIANGLES:
        for (u32 i = 0; i + 2 < count; i += 3) emit(st, first + i, first + i + 1, first + i + 2);
        break;
    case RSX_PRIMITIVE_TRIANGLE_STRIP:
        for (u32 i = 0; i + 2 < count; i++)
            emit(st, first + i, first + i + 1 + (i & 1), first + i + 2 - (i & 1));
        break;
    case RSX_PRIMITIVE_TRIANGLE_FAN:
    case RSX_PRIMITIVE_POLYGON:
        for (u32 i = 1; i + 1 < count; i++) emit(st, first, first + i, first + i + 1);
        break;
    case RSX_PRIMITIVE_QUADS:
        for (u32 i = 0; i + 3 < count; i += 4) {
            emit(st, first + i, first + i + 1, first + i + 2);
            emit(st, first + i, first + i + 2, first + i + 3);
        }
        break;
    case RSX_PRIMITIVE_QUAD_STRIP:
        for (u32 i = 0; i + 3 < count; i += 2) {
            emit(st, first + i, first + i + 1, first + i + 3);
            emit(st, first + i, first + i + 3, first + i + 2);
        }
        break;
    default:
        break;   /* points and lines: no coverage, as in the null backend */
    }
}

static int vk_ensure_vertex_buffer(VkDeviceSize bytes)
{
    if (s_vk.vertices.buf && s_vk.vertices.size >= bytes) return 0;
    VkDeviceSize size = s_vk.vertices.size ? s_vk.vertices.size : 64 * 1024;
    while (size < bytes) size *= 2;
    vk_destroy_buffer(&s_vk.vertices);   /* idle: every submission is waited on */
    if (vk_create_host_buffer(&s_vk.vertices, size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) return -1;
    if (!s_vk.vertices.coherent) { VK_LOG("vertex memory is not host-coherent\n"); return -1; }
    return 0;
}

/* ---------------------------------------------------------------------------
 * Guest programs (stage V3)
 *
 * With the HLSL -> SPIR-V translator built and PS3RECOMP_VK_GUEST_PROGRAMS=1,
 * a draw runs the guest's own vertex and fragment programs, as the Metal
 * backend's guest path does and keyed the same way: decompiled to HLSL by the
 * shared decompilers, translated by rsx_shader_spirv.cpp, cached per program.
 * Constants use the Metal/D3D12 layout: VPConst is the 512 transform
 * constants plus the viewport epilogue, PSConstants the fragment program's
 * inline constants plus fp_alpha.
 *
 * Vertices are still fetched and expanded on the CPU through the shared
 * rsx_fetch_attrib(), now all sixteen attributes (a disabled one reads its
 * constant register, as on the hardware). Every RSX vertex format therefore
 * arrives as float4, and the pipeline has one fixed input layout: location n
 * is attribute n, which is what the vertex decompiler declares.
 *
 * A program that needs a cube map or a vertex texture falls back to the
 * fixed-function path, with a message. Not honoured on this path yet: the RSX
 * viewport and scissor (the whole target is used, as on the fallback path),
 * blending, stencil, culling, per-unit sampler state.
 * -------------------------------------------------------------------------*/
#define VK_VP_CACHE      256
#define VK_FP_CACHE      512
#define VK_GPIPE_CACHE   512
#define VK_FP_MAX_BYTES  4096u              /* bound on a fragment program, as Metal/D3D12 */
#define VK_HLSL_BYTES    (256u * 1024u)
#define VK_SPV_WORDS     (256u * 1024u)
#define VK_VPCONST_BYTES ((RSX_MAX_VERTEX_CONSTANTS + 2u) * 16u)
#define VK_GUEST_ATTRS   16u
#define VK_GUEST_FLOATS  (VK_GUEST_ATTRS * 4u)   /* per vertex */

typedef struct { u32 hash; VkShaderModule mod; } vk_vp_entry;
typedef struct { u64 key; VkShaderModule mod; u32 nconst; } vk_fp_entry;
typedef struct { int vs, fs; u32 depth_key; VkPipeline pipe; } vk_gpipe_entry;

static struct {
    int                   on;
    char*                 hlsl;
    u32*                  spv;
    char                  log[4096];
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout      pipe_layout;
    VkDescriptorPool      pool;
    VkDescriptorSet       set;
    vk_buffer             ubo;              /* VPConst at 0, PSConstants at ps_off */
    VkDeviceSize          ps_off;
    VkDeviceSize          max_ubo_range;
    vk_buffer             vertices;
    float*                cpu;
    size_t                cpu_cap, cpu_count;   /* in vertices */
    vk_vp_entry           vp[VK_VP_CACHE];     u32 vp_count;
    vk_fp_entry           fp[VK_FP_CACHE];     u32 fp_count;
    vk_gpipe_entry        pipes[VK_GPIPE_CACHE]; u32 pipe_count;
    rsx_fp_constant_block consts;
    u32                   draws, last_draws;
    u32                   warned;
} s_g;

int rsx_vulkan_backend_guest_programs(void)
{
    const char* v = getenv("PS3RECOMP_VK_GUEST_PROGRAMS");
    return rsx_hlsl_to_spirv_available() && v && v[0] && v[0] != '0';
}

static u32 vk_fnv1a32(const u8* p, u32 n)
{
    u32 h = 2166136261u;
    for (u32 i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

static u64 vk_fnv1a64(const void* d, size_t n, u64 h)
{
    const u8* p = (const u8*)d;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

/* Built from the registers, as the Metal and D3D12 backends do. */
static u32 vk_cube_mask(const rsx_state* st)
{
    u32 m = 0;
    for (u32 u = 0; u < RSX_MAX_TEXTURES; u++) {
        const rsx_texture_state* t = &st->textures[u];
        const u32 w = (t->image_rect >> 16) & 0xFFFFu, h = t->image_rect & 0xFFFFu;
        if ((t->control0 & 0x80000000u) && (t->format & 4u) && w && w == h) m |= 1u << u;
    }
    return m;
}

static u32 vk_vtex_mask(const rsx_state* st)
{
    u32 m = 0;
    for (u32 u = 0; u < RSX_MAX_VERTEX_TEXTURES; u++)
        if (st->vertex_textures[u].control0 & 0x80000000u) m |= 1u << u;
    return m;
}

static VkShaderModule vk_guest_module(const char* what, int stage)
{
    u32 n = 0;
    if (rsx_hlsl_to_spirv(s_g.hlsl, stage, s_g.spv, VK_SPV_WORDS, &n,
                          s_g.log, sizeof s_g.log) != 0) {
        VK_LOG("%s: %s\n", what, s_g.log);
        return VK_NULL_HANDLE;
    }
    VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                    .codeSize = (size_t)n * 4u, .pCode = s_g.spv };
    VkShaderModule m = VK_NULL_HANDLE;
    if (pvkCreateShaderModule(s_vk.device, &ci, NULL, &m) != VK_SUCCESS) {
        VK_LOG("%s: vkCreateShaderModule failed\n", what);
        return VK_NULL_HANDLE;
    }
    return m;
}

/* The vertex program, keyed on its own bytes to its end bit (as Metal). */
static int vk_vp_slot(const rsx_state* st)
{
    if (st->vp_ucode_bytes < 16) return -1;
    u32 vstart = st->transform_program_start * 16u;
    if (vstart >= st->vp_ucode_bytes) vstart = 0;
    const u8* uc = st->vp_ucode + vstart;
    const u32 avail = st->vp_ucode_bytes - vstart;
    const u32 instrs = rsx_vp_program_size_instrs(uc, avail);
    const u32 len = instrs ? instrs * 16u : avail;
    const u32 hash = vk_fnv1a32(uc, len);
    for (u32 i = 0; i < s_g.vp_count; i++)
        if (s_g.vp[i].hash == hash) return s_g.vp[i].mod ? (int)i : -1;
    if (s_g.vp_count >= VK_VP_CACHE) return -1;

    char what[64];
    snprintf(what, sizeof what, "vertex program %08X", hash);
    VkShaderModule m = VK_NULL_HANDLE;
    const int ni = rsx_vp_decompile_ex(uc, len, 0, s_g.hlsl, VK_HLSL_BYTES);
    if (ni <= 0) VK_LOG("%s: decompile failed (%d)\n", what, ni);
    else         m = vk_guest_module(what, RSX_SHADER_STAGE_VERTEX);
    if (s_g.vp_count < 32)
        VK_LOG("%s: %d instrs -> %s\n", what, ni, m ? "ok" : "FAILED (fixed-function instead)");
    const int slot = (int)s_g.vp_count++;
    s_g.vp[slot].hash = hash;
    s_g.vp[slot].mod  = m;
    return m ? slot : -1;
}

/* The fragment program, resolved and keyed as the Metal backend's
 * fp_slot_for: structure (inline constants live in PSConstants), export
 * width, and the alpha test patched into the source. No cube units here --
 * the caller has already sent those to the fixed-function path. */
static int vk_fp_slot(const rsx_state* st, const u8** uc)
{
    if (!vm_base || st->shader_program == 0) return -1;
    const u32 off = cellGcmResolveLocated((st->shader_program & 3u) == 1u,
                                          st->shader_program & ~3u);
    if (off == 0xFFFFFFFFu) return -1;
    *uc = vm_base + off;
    u64 key = rsx_fp_structural_hash(*uc, VK_FP_MAX_BYTES, 1469598103934665603ull);
    if (!key) return -1;                       /* malformed or unterminated */
    const u32 ctrl     = st->shader_control;
    const u32 ctrl_key = ctrl & CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS;
    const u32 alpha_en = (st->alpha_test_enable && st->alpha_func != 0x0207u) ? 1u : 0u;
    const u32 alpha_fn = alpha_en ? st->alpha_func : 0u;
    key = vk_fnv1a64(&ctrl_key, sizeof ctrl_key, key);
    key = vk_fnv1a64(&alpha_en, sizeof alpha_en, key);
    key = vk_fnv1a64(&alpha_fn, sizeof alpha_fn, key);
    for (u32 i = 0; i < s_g.fp_count; i++)
        if (s_g.fp[i].key == key) return s_g.fp[i].mod ? (int)i : -1;
    if (s_g.fp_count >= VK_FP_CACHE) return -1;

    char what[64];
    snprintf(what, sizeof what, "fragment program %016llX", (unsigned long long)key);
    VkShaderModule m = VK_NULL_HANDLE;
    u32 nconst = 0;
    int ni = rsx_fp_decompile_buffered_ex(*uc, VK_FP_MAX_BYTES, ctrl, 0,
                                          s_g.hlsl, VK_HLSL_BYTES, &nconst);
    if (ni > 0 && alpha_en &&
        rsx_fp_apply_alpha_test_buffered(s_g.hlsl, VK_HLSL_BYTES, st->alpha_func) < 0)
        ni = -1;
    if (ni <= 0) VK_LOG("%s: decompile failed (%d)\n", what, ni);
    else         m = vk_guest_module(what, RSX_SHADER_STAGE_FRAGMENT);
    if (s_g.fp_count < 32)
        VK_LOG("%s: %d instrs, %u constants -> %s\n", what, ni, nconst,
               m ? "ok" : "FAILED (fixed-function instead)");
    const int slot = (int)s_g.fp_count++;
    s_g.fp[slot].key    = key;
    s_g.fp[slot].mod    = m;
    s_g.fp[slot].nconst = nconst;
    return m ? slot : -1;
}

static VkPipeline vk_guest_pipeline(int vs, int fs, int ztest, int zwrite, u32 cmp)
{
    const u32 dkey = (u32)ztest | ((u32)zwrite << 1) | (cmp << 2);
    for (u32 i = 0; i < s_g.pipe_count; i++)
        if (s_g.pipes[i].vs == vs && s_g.pipes[i].fs == fs && s_g.pipes[i].depth_key == dkey)
            return s_g.pipes[i].pipe;
    if (s_g.pipe_count >= VK_GPIPE_CACHE) return VK_NULL_HANDLE;

    VkPipelineShaderStageCreateInfo stages[2] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = s_g.vp[vs].mod, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = s_g.fp[fs].mod, .pName = "main" },
    };
    VkVertexInputBindingDescription vb = { 0, VK_GUEST_FLOATS * 4u, VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription va[VK_GUEST_ATTRS];
    for (u32 a = 0; a < VK_GUEST_ATTRS; a++) {
        va[a].location = a;
        va[a].binding  = 0;
        va[a].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        va[a].offset   = a * 16u;
    }
    VkPipelineVertexInputStateCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vb,
        .vertexAttributeDescriptionCount = VK_GUEST_ATTRS, .pVertexAttributeDescriptions = va };
    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkPipelineViewportStateCreateInfo vp = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1 };
    VkPipelineRasterizationStateCreateInfo rs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkPipelineDepthStencilStateCreateInfo ds = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = ztest ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = zwrite ? VK_TRUE : VK_FALSE,
        .depthCompareOp = (VkCompareOp)cmp };
    VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xF };
    VkPipelineColorBlendStateCreateInfo cb = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &cba };
    VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2, .pDynamicStates = dyn };
    VkGraphicsPipelineCreateInfo gci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = stages, .pVertexInputState = &vi,
        .pInputAssemblyState = &ia, .pViewportState = &vp, .pRasterizationState = &rs,
        .pMultisampleState = &ms, .pDepthStencilState = &ds, .pColorBlendState = &cb,
        .pDynamicState = &dy, .layout = s_g.pipe_layout, .renderPass = s_vk.render_pass,
        .subpass = 0 };
    VkPipeline p = VK_NULL_HANDLE;
    VkResult r = pvkCreateGraphicsPipelines(s_vk.device, VK_NULL_HANDLE, 1, &gci, NULL, &p);
    if (r != VK_SUCCESS) {
        VK_LOG("guest pipeline: vkCreateGraphicsPipelines failed (%d)\n", (int)r);
        return VK_NULL_HANDLE;
    }
    vk_gpipe_entry* e = &s_g.pipes[s_g.pipe_count++];
    e->vs = vs; e->fs = fs; e->depth_key = dkey; e->pipe = p;
    return p;
}

/* One vertex: all sixteen attributes as float4, attribute n at float 4n. */
static int vk_guest_push(const rsx_state* st, u32 idx)
{
    if (s_g.cpu_count == s_g.cpu_cap) {
        size_t cap = s_g.cpu_cap ? s_g.cpu_cap * 2 : 1024;
        float* n = (float*)realloc(s_g.cpu, cap * VK_GUEST_FLOATS * sizeof(float));
        if (!n) return -1;
        s_g.cpu = n; s_g.cpu_cap = cap;
    }
    float* v = s_g.cpu + s_g.cpu_count * VK_GUEST_FLOATS;
    for (u32 a = 0; a < VK_GUEST_ATTRS; a++) rsx_fetch_attrib(st, (int)a, idx, v + a * 4u);
    s_g.cpu_count++;
    return 0;
}

static void vk_guest_tri(const rsx_state* st, u32 i0, u32 i1, u32 i2)
{
    vk_guest_push(st, i0);
    vk_guest_push(st, i1);
    vk_guest_push(st, i2);
}

static int vk_guest_init(void)
{
    if (!rsx_vulkan_backend_guest_programs()) return 0;
    s_g.hlsl = (char*)malloc(VK_HLSL_BYTES);
    s_g.spv  = (u32*)malloc(VK_SPV_WORDS * sizeof(u32));
    if (!s_g.hlsl || !s_g.spv) { VK_LOG("guest programs: out of memory\n"); return -1; }

    VkPhysicalDeviceProperties p;
    pvkGetPhysicalDeviceProperties(s_vk.phys, &p);
    s_g.max_ubo_range = p.limits.maxUniformBufferRange;
    VkDeviceSize align = p.limits.minUniformBufferOffsetAlignment;
    if (align == 0) align = 256;
    s_g.ps_off = (VK_VPCONST_BYTES + align - 1u) / align * align;
    if (VK_VPCONST_BYTES > s_g.max_ubo_range) {
        VK_LOG("guest programs: maxUniformBufferRange %llu < VPConst\n",
               (unsigned long long)s_g.max_ubo_range);
        return -1;
    }

    const VkShaderStageFlags both = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding b[4] = {
        { RSX_SPIRV_VPCONST_BINDING, VK_UB_TYPE, 1, both, NULL },
        { RSX_SPIRV_PSCONST_BINDING, VK_UB_TYPE, 1, both, NULL },
        { RSX_SPIRV_TEXTURE_BINDING, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, RSX_MAX_TEXTURES,
          VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { RSX_SPIRV_SAMPLER_BINDING, VK_DESCRIPTOR_TYPE_SAMPLER, RSX_MAX_TEXTURES,
          VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
    };
    VkDescriptorSetLayoutCreateInfo lci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4, .pBindings = b };
    VK_CHECK(pvkCreateDescriptorSetLayout(s_vk.device, &lci, NULL, &s_g.set_layout),
             "vkCreateDescriptorSetLayout(guest)");
    VkDescriptorPoolSize ps[3] = {
        { VK_UB_TYPE, 2 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, RSX_MAX_TEXTURES },
        { VK_DESCRIPTOR_TYPE_SAMPLER, RSX_MAX_TEXTURES },
    };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = 1, .poolSizeCount = 3, .pPoolSizes = ps };
    VK_CHECK(pvkCreateDescriptorPool(s_vk.device, &pci, NULL, &s_g.pool),
             "vkCreateDescriptorPool(guest)");
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = s_g.pool, .descriptorSetCount = 1,
                                        .pSetLayouts = &s_g.set_layout };
    VK_CHECK(pvkAllocateDescriptorSets(s_vk.device, &dai, &s_g.set), "vkAllocateDescriptorSets(guest)");
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1, .pSetLayouts = &s_g.set_layout };
    VK_CHECK(pvkCreatePipelineLayout(s_vk.device, &plci, NULL, &s_g.pipe_layout),
             "vkCreatePipelineLayout(guest)");
    if (vk_create_host_buffer(&s_g.ubo,
                              s_g.ps_off + (RSX_FP_MAX_INLINE_CONSTANTS + 1u) * 16u,
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) || !s_g.ubo.coherent) {
        VK_LOG("guest programs: no host-coherent uniform memory\n");
        return -1;
    }
    s_g.on = 1;
    VK_LOG("guest programs: on (HLSL -> SPIR-V through glslang)\n");
    return 0;
}

static void vk_guest_shutdown(void)
{
    if (s_vk.device) {
        for (u32 i = 0; i < s_g.pipe_count; i++)
            if (s_g.pipes[i].pipe) pvkDestroyPipeline(s_vk.device, s_g.pipes[i].pipe, NULL);
        for (u32 i = 0; i < s_g.vp_count; i++)
            if (s_g.vp[i].mod) pvkDestroyShaderModule(s_vk.device, s_g.vp[i].mod, NULL);
        for (u32 i = 0; i < s_g.fp_count; i++)
            if (s_g.fp[i].mod) pvkDestroyShaderModule(s_vk.device, s_g.fp[i].mod, NULL);
        if (s_g.pipe_layout) pvkDestroyPipelineLayout(s_vk.device, s_g.pipe_layout, NULL);
        if (s_g.pool)        pvkDestroyDescriptorPool(s_vk.device, s_g.pool, NULL);
        if (s_g.set_layout)  pvkDestroyDescriptorSetLayout(s_vk.device, s_g.set_layout, NULL);
        vk_destroy_buffer(&s_g.ubo);
        vk_destroy_buffer(&s_g.vertices);
    }
    free(s_g.hlsl);
    free(s_g.spv);
    free(s_g.cpu);
    memset(&s_g, 0, sizeof s_g);
}

static int vk_ensure_buffer(vk_buffer* b, VkDeviceSize bytes, VkBufferUsageFlags usage)
{
    if (b->buf && b->size >= bytes) return 0;
    VkDeviceSize size = b->size ? b->size : 64 * 1024;
    while (size < bytes) size *= 2;
    vk_destroy_buffer(b);                 /* idle: every submission is waited on */
    if (vk_create_host_buffer(b, size, usage)) return -1;
    if (!b->coherent) { VK_LOG("vertex memory is not host-coherent\n"); return -1; }
    return 0;
}


/* Returns 1 when the draw went through the guest programs (or had nothing to
 * draw), 0 when the caller must use the fixed-function path instead. */
static int vk_guest_draw(const rsx_state* st, u32 prim, u32 first, u32 count)
{
    if (!s_g.on) return 0;
    if (vk_cube_mask(st) || vk_vtex_mask(st)) {
        if (!(s_g.warned & 1u)) {
            VK_LOG("guest programs: cube maps and vertex textures are not supported "
                   "yet; those draws use the fixed-function path\n");
            s_g.warned |= 1u;
        }
        return 0;
    }
    const int vs = vk_vp_slot(st);
    if (vs < 0) return 0;
    const u8* fp_uc = NULL;
    const int fs = vk_fp_slot(st, &fp_uc);
    if (fs < 0) return 0;
    /* The constant count must be what the program was compiled against. */
    if (rsx_fp_collect_constants(fp_uc, VK_FP_MAX_BYTES, &s_g.consts) < 0 ||
        s_g.consts.count != s_g.fp[fs].nconst)
        return 0;
    const u32 nslots = s_g.consts.count ? s_g.consts.count : 1u;
    const u32 fp_len = (nslots + 1u) * 16u;
    if (fp_len > s_g.max_ubo_range) return 0;

    const int ztest  = st->depth_test_enable ? 1 : 0;
    const int zwrite = ztest && st->depth_mask;
    VkPipeline pipe = vk_guest_pipeline(vs, fs, ztest, zwrite, vk_compare_slot(st->depth_func));
    if (!pipe) return 0;

    s_g.cpu_count = 0;
    vk_expand(st, prim, first, count, vk_guest_tri);
    if (!s_g.cpu_count) return 1;
    const VkDeviceSize bytes = (VkDeviceSize)s_g.cpu_count * VK_GUEST_FLOATS * sizeof(float);
    if (vk_ensure_buffer(&s_g.vertices, bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) return 0;
    memcpy(s_g.vertices.ptr, s_g.cpu, (size_t)bytes);

    /* VPConst: the 512 transform constants, then the viewport epilogue as the
     * Metal and D3D12 backends write it -- x/y identity, and z remapped from
     * GL clip space to [0, 1] when the guest programmed a z scale. */
    u8* vpc = (u8*)s_g.ubo.ptr;
    memcpy(vpc, st->vertex_constants, RSX_MAX_VERTEX_CONSTANTS * 16u);
    float* vpx = (float*)(vpc + RSX_MAX_VERTEX_CONSTANTS * 16u);
    vpx[0] = vpx[1] = vpx[3] = 1.0f;
    vpx[4] = vpx[5] = vpx[7] = 0.0f;
    if (st->viewport_scale[2] != 0.0f) { vpx[2] = st->viewport_scale[2]; vpx[6] = st->viewport_offset[2]; }
    else                                { vpx[2] = 1.0f;                  vpx[6] = 0.0f; }
    /* PSConstants: inline constants as host-order bit patterns, then fp_alpha. */
    u8* fpc = (u8*)s_g.ubo.ptr + s_g.ps_off;
    memset(fpc, 0, fp_len);
    if (s_g.consts.count) memcpy(fpc, s_g.consts.values, s_g.consts.count * 16u);
    float* alpha = (float*)(fpc + nslots * 16u);
    alpha[0] = rsx_fp_alpha_ref(st->alpha_ref, st->surface_format & 0x1Fu);

    /* Descriptors: rewritten per draw, which is safe because every submission
     * is waited on before the next one is recorded. */
    VkDescriptorBufferInfo bi[2] = {
        { s_g.ubo.buf, 0, VK_VPCONST_BYTES },
        { s_g.ubo.buf, s_g.ps_off, fp_len },
    };
    VkDescriptorImageInfo ii[RSX_MAX_TEXTURES], si[RSX_MAX_TEXTURES];
    for (u32 u = 0; u < RSX_MAX_TEXTURES; u++) {
        ii[u] = (VkDescriptorImageInfo){ VK_NULL_HANDLE,
                                         s_vk.tex_ready[u] ? s_vk.tex[u].view : s_vk.dummy.view,
                                         VK_IMAGE_LAYOUT_GENERAL };
        si[u] = (VkDescriptorImageInfo){ s_vk.tex_sampler[u] ? s_vk.tex_sampler[u] : s_vk.sampler,
                                         VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
    }
    VkWriteDescriptorSet w[4] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s_g.set,
          .dstBinding = RSX_SPIRV_VPCONST_BINDING, .descriptorCount = 1,
          .descriptorType = VK_UB_TYPE, .pBufferInfo = &bi[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s_g.set,
          .dstBinding = RSX_SPIRV_PSCONST_BINDING, .descriptorCount = 1,
          .descriptorType = VK_UB_TYPE, .pBufferInfo = &bi[1] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s_g.set,
          .dstBinding = RSX_SPIRV_TEXTURE_BINDING, .descriptorCount = RSX_MAX_TEXTURES,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = ii },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s_g.set,
          .dstBinding = RSX_SPIRV_SAMPLER_BINDING, .descriptorCount = RSX_MAX_TEXTURES,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .pImageInfo = si },
    };
    pvkUpdateDescriptorSets(s_vk.device, 4, w, 0, NULL);

    if (vk_begin()) return 0;
    vk_barrier_image(s_vk.color.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    vk_barrier_image(s_vk.depth.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
    VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                  .renderPass = s_vk.render_pass, .framebuffer = s_vk.framebuffer,
                                  .renderArea = { { 0, 0 }, { s_vk.width, s_vk.height } } };
    pvkCmdBeginRenderPass(s_vk.cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    pvkCmdBindPipeline(s_vk.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    VkViewport vpt = { 0.0f, 0.0f, (float)s_vk.width, (float)s_vk.height, 0.0f, 1.0f };
    VkRect2D sc = { { 0, 0 }, { s_vk.width, s_vk.height } };
    pvkCmdSetViewport(s_vk.cmd, 0, 1, &vpt);
    pvkCmdSetScissor(s_vk.cmd, 0, 1, &sc);
    s_rec_epoch++;
    { static const uint32_t zero2[2] = { 0, 0 };
      pvkCmdBindDescriptorSets(s_vk.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_g.pipe_layout,
                               0, 1, &s_g.set, vk_ub_dynamic() ? 2u : 0u, zero2); }
    VkDeviceSize off = 0;
    pvkCmdBindVertexBuffers(s_vk.cmd, 0, 1, &s_g.vertices.buf, &off);
    pvkCmdDraw(s_vk.cmd, (u32)s_g.cpu_count, 1, 0, 0);
    pvkCmdEndRenderPass(s_vk.cmd);
    if (vk_submit_and_wait()) return 0;
    s_g.draws++;
    return 1;
}

static void vk_draw(u32 prim, u32 first, u32 count)
{
    const rsx_state* st = s_vk.state;
    if (!st || !s_vk.device || count < 3) return;
    if (vk_guest_draw(st, prim, first, count)) return;

    s_vk.cpu_count = 0;
    vk_expand(st, prim, first, count, vk_emit_tri);
    if (!s_vk.cpu_count) return;

    const VkDeviceSize bytes = (VkDeviceSize)s_vk.cpu_count * sizeof(vk_vertex);
    if (vk_ensure_vertex_buffer(bytes)) return;
    memcpy(s_vk.vertices.ptr, s_vk.cpu_verts, (size_t)bytes);

    /* Depth writes need the test enabled, as on the hardware and in both
     * reference backends. */
    const int ztest  = st->depth_test_enable ? 1 : 0;
    const int zwrite = ztest && st->depth_mask;
    VkPipeline pipe = vk_get_pipeline(ztest, zwrite, vk_compare_slot(st->depth_func));
    if (!pipe) return;
    const int32_t textured = (s_vk.tex_ready[0] && st->vertex_attribs[8].enabled) ? 1 : 0;

    if (vk_begin()) return;
    vk_barrier_image(s_vk.color.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    vk_barrier_image(s_vk.depth.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
    VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                  .renderPass = s_vk.render_pass, .framebuffer = s_vk.framebuffer,
                                  .renderArea = { { 0, 0 }, { s_vk.width, s_vk.height } } };
    pvkCmdBeginRenderPass(s_vk.cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    pvkCmdBindPipeline(s_vk.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    /* Whole target, as the null backend's ndc_to_px maps it. */
    VkViewport vp = { 0.0f, 0.0f, (float)s_vk.width, (float)s_vk.height, 0.0f, 1.0f };
    VkRect2D sc = { { 0, 0 }, { s_vk.width, s_vk.height } };
    pvkCmdSetViewport(s_vk.cmd, 0, 1, &vp);
    pvkCmdSetScissor(s_vk.cmd, 0, 1, &sc);
    s_rec_epoch++;
    pvkCmdBindDescriptorSets(s_vk.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_vk.pipe_layout,
                             0, 1, &s_vk.desc_set, 0, NULL);
    pvkCmdPushConstants(s_vk.cmd, s_vk.pipe_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                        0, sizeof textured, &textured);
    VkDeviceSize off = 0;
    pvkCmdBindVertexBuffers(s_vk.cmd, 0, 1, &s_vk.vertices.buf, &off);
    pvkCmdDraw(s_vk.cmd, (u32)s_vk.cpu_count, 1, 0, 0);
    pvkCmdEndRenderPass(s_vk.cmd);
    vk_submit_and_wait();
}

static void vk_cb_draw_arrays(void* ud, u32 primitive, u32 first, u32 count)
{
    (void)ud;
    vk_draw(primitive, first, count);
}

static void vk_cb_draw_indexed(void* ud, u32 primitive, u32 offset, u32 count)
{
    /* Not read yet, as in the null backend: the harness geometry is
     * non-indexed. Real titles need this; it lands with the index fetch. */
    (void)ud; (void)primitive; (void)offset; (void)count;
}

/* ---------------------------------------------------------------------------
 * Clear and present
 * -------------------------------------------------------------------------*/
static void vk_cb_clear(void* ud, u32 flags, u32 color, float depth, u8 stencil)
{
    (void)ud; (void)stencil;                       /* no stencil buffer yet */
    s_vk.clear_argb = color;
    /* CLEAR_SURFACE bits: 0x01 depth, 0x02 stencil, 0xF0 the colour channels
     * (all or nothing, as in the null backend). */
    const int do_color = (flags & 0xF0u) != 0;
    const int do_depth = (flags & 0x01u) != 0;
    if (!s_vk.device || (!do_color && !do_depth)) return;

    if (vk_begin()) return;
    if (do_color) {
        VkClearColorValue v;
        v.float32[0] = (float)((color >> 16) & 0xFFu) / 255.0f;   /* R */
        v.float32[1] = (float)((color >>  8) & 0xFFu) / 255.0f;   /* G */
        v.float32[2] = (float)( color        & 0xFFu) / 255.0f;   /* B */
        v.float32[3] = (float)((color >> 24) & 0xFFu) / 255.0f;   /* A */
        VkImageSubresourceRange r = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_barrier_image(s_vk.color.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
        pvkCmdClearColorImage(s_vk.cmd, s_vk.color.img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
    }
    if (do_depth) {
        VkClearDepthStencilValue d = { depth, 0 };
        VkImageSubresourceRange r = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        vk_barrier_image(s_vk.depth.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
        pvkCmdClearDepthStencilImage(s_vk.cmd, s_vk.depth.img, VK_IMAGE_LAYOUT_GENERAL, &d, 1, &r);
    }
    vk_submit_and_wait();
}

/* ---------------------------------------------------------------------------
 * Optional window: PS3RECOMP_VK_WINDOW=1
 *
 * An SDL2 window with a Vulkan swapchain. Rendering still targets the
 * offscreen image, so readback and every test are unchanged; a present then
 * also blits that image into the next swapchain image. Any failure on this
 * path drops back to headless with a message instead of failing the run.
 * -------------------------------------------------------------------------*/
static int vk_env_on(const char* name)
{
    const char* v = getenv(name);
    return v && v[0] && v[0] != '0';
}

static void vk_window_drop(const char* why)
{
    if (why) VK_LOG("window disabled, continuing headless: %s\n", why);
    if (s_vk.device && pvkDeviceWaitIdle) {
        vk_device_idle();
        for (u32 i = 0; i < 8; i++)
            if (s_vk.sc_done[i]) {
                pvkDestroySemaphore(s_vk.device, s_vk.sc_done[i], NULL);
                s_vk.sc_done[i] = VK_NULL_HANDLE;
            }
        if (s_vk.sem_acquire) pvkDestroySemaphore(s_vk.device, s_vk.sem_acquire, NULL);
        if (s_vk.swapchain && pvkDestroySwapchainKHR)
            pvkDestroySwapchainKHR(s_vk.device, s_vk.swapchain, NULL);
    }
    if (s_vk.surface && s_vk.instance && pvkDestroySurfaceKHR)
        pvkDestroySurfaceKHR(s_vk.instance, s_vk.surface, NULL);
    if (s_vk.window) {
        SDL_DestroyWindow(s_vk.window);
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
    }
    s_vk.window = NULL;
    s_vk.surface = VK_NULL_HANDLE;
    s_vk.swapchain = VK_NULL_HANDLE;
    s_vk.sem_acquire = VK_NULL_HANDLE;
    s_vk.sc_count = 0;
    s_vk.windowed = 0;
}

static void vk_window_open(const char* title)
{
    if (!vk_env_on("PS3RECOMP_VK_WINDOW")) return;
    if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
        VK_LOG("window disabled, continuing headless: SDL video init failed: %s\n", SDL_GetError());
        return;
    }
    /* The controller subsystem is initialised here, on the thread that pumps SDL's events: cellPad
     * used to do it from the guest thread while this one was inside SDL_WaitEventTimeout, and the
     * two raced in SDL's joystick enumeration (malloc "pointer being freed was not allocated"). */
    if (SDL_WasInit(SDL_INIT_GAMECONTROLLER) == 0) SDL_InitSubSystem(SDL_INIT_GAMECONTROLLER);
    Uint32 flags = SDL_WINDOW_VULKAN | SDL_WINDOW_SHOWN;
    if (vk_env_on("PS3RECOMP_VK_FULLSCREEN")) flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
    /* PS3RECOMP_VK_HIDPI=1: a drawable in the monitor's own pixels (2x on Retina) rather than the
     * window's points, which macOS would enlarge with a plain bilinear filter; the upscaler
     * (PS3RECOMP_VK_UPSCALE) then fills it. */
    if (vk_env_on("PS3RECOMP_VK_HIDPI")) flags |= SDL_WINDOW_ALLOW_HIGHDPI;
    s_vk.window = SDL_CreateWindow(title && title[0] ? title : "ps3recomp",
                                   SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                   (int)s_vk.width, (int)s_vk.height, flags);
    s_vk.windowed = 1;          /* so vk_window_drop also shuts SDL video down */
    if (!s_vk.window) {
        char why[256];
        snprintf(why, sizeof why, "SDL_CreateWindow failed: %s", SDL_GetError());
        vk_window_drop(why);
        return;
    }
    unsigned n = 0;
    if (!SDL_Vulkan_GetInstanceExtensions(s_vk.window, &n, NULL) || n == 0 || n > 16 ||
        !SDL_Vulkan_GetInstanceExtensions(s_vk.window, &n, s_vk.inst_exts)) {
        char why[256];
        snprintf(why, sizeof why, "no Vulkan surface extensions (%s)", SDL_GetError());
        vk_window_drop(why);
        return;
    }
    s_vk.inst_ext_count = n;
    const char* hold = getenv("PS3RECOMP_VK_HOLD");
    s_vk.hold_seconds = hold ? atof(hold) : 0.0;
    VK_LOG("window: SDL video driver '%s', %u surface extension(s)\n",
           SDL_GetCurrentVideoDriver(), n);
}

static int vk_create_swapchain(void)
{
    VkSurfaceCapabilitiesKHR caps;
    VK_CHECK(pvkGetPhysicalDeviceSurfaceCapabilitiesKHR(s_vk.phys, s_vk.surface, &caps),
             "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        VK_LOG("swapchain images cannot be blit destinations\n"); return -1;
    }

    u32 nf = 0;
    VK_CHECK(pvkGetPhysicalDeviceSurfaceFormatsKHR(s_vk.phys, s_vk.surface, &nf, NULL),
             "vkGetPhysicalDeviceSurfaceFormatsKHR");
    if (nf == 0) { VK_LOG("surface reports no formats\n"); return -1; }
    if (nf > 64) nf = 64;
    VkSurfaceFormatKHR fmts[64];
    VK_CHECK(pvkGetPhysicalDeviceSurfaceFormatsKHR(s_vk.phys, s_vk.surface, &nf, fmts),
             "vkGetPhysicalDeviceSurfaceFormatsKHR");
    /* UNORM, not SRGB: the offscreen target holds the guest's values as-is. */
    VkSurfaceFormatKHR fmt = fmts[0];
    if (nf == 1 && fmts[0].format == VK_FORMAT_UNDEFINED) {
        fmt.format = VK_FORMAT_B8G8R8A8_UNORM;
    } else {
        static const VkFormat pref[] = { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM };
        int found = 0;
        for (size_t k = 0; k < 2 && !found; k++)
            for (u32 i = 0; i < nf; i++)
                if (fmts[i].format == pref[k]) { fmt = fmts[i]; found = 1; break; }
    }
    VkFormatProperties fp;
    pvkGetPhysicalDeviceFormatProperties(s_vk.phys, fmt.format, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
        VK_LOG("swapchain format %d cannot be a blit destination\n", (int)fmt.format); return -1;
    }

    VkExtent2D ext = caps.currentExtent;
    if (ext.width == 0xFFFFFFFFu) {
        int w = 0, hh = 0;
        SDL_Vulkan_GetDrawableSize(s_vk.window, &w, &hh);
        ext.width = (u32)w; ext.height = (u32)hh;
        if (ext.width  < caps.minImageExtent.width)  ext.width  = caps.minImageExtent.width;
        if (ext.height < caps.minImageExtent.height) ext.height = caps.minImageExtent.height;
        if (ext.width  > caps.maxImageExtent.width)  ext.width  = caps.maxImageExtent.width;
        if (ext.height > caps.maxImageExtent.height) ext.height = caps.maxImageExtent.height;
    }
    if (!ext.width || !ext.height) { VK_LOG("window has no drawable area\n"); return -1; }

    u32 count = caps.minImageCount + 1;
    if (caps.maxImageCount && count > caps.maxImageCount) count = caps.maxImageCount;

    static const VkCompositeAlphaFlagBitsKHR alphas[] = {
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
    };
    VkCompositeAlphaFlagBitsKHR alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    for (size_t k = 0; k < 4; k++)
        if (caps.supportedCompositeAlpha & alphas[k]) { alpha = alphas[k]; break; }

    VkSwapchainKHR old = s_vk.swapchain;
    VkSwapchainCreateInfoKHR sci = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = s_vk.surface, .minImageCount = count,
        .imageFormat = fmt.format, .imageColorSpace = fmt.colorSpace,
        .imageExtent = ext, .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = caps.currentTransform, .compositeAlpha = alpha,
        .presentMode = VK_PRESENT_MODE_FIFO_KHR,   /* the one mode every driver has */
        .clipped = VK_TRUE, .oldSwapchain = old,
    };
    VkSwapchainKHR sc = VK_NULL_HANDLE;
    VkResult r = pvkCreateSwapchainKHR(s_vk.device, &sci, NULL, &sc);
    if (old) pvkDestroySwapchainKHR(s_vk.device, old, NULL);
    s_vk.swapchain = VK_NULL_HANDLE;
    if (r != VK_SUCCESS) { VK_LOG("vkCreateSwapchainKHR failed (VkResult %d)\n", (int)r); return -1; }
    s_vk.swapchain = sc;

    u32 n = 0;
    VK_CHECK(pvkGetSwapchainImagesKHR(s_vk.device, sc, &n, NULL), "vkGetSwapchainImagesKHR");
    if (n == 0 || n > 8) { VK_LOG("unsupported swapchain image count %u\n", n); return -1; }
    VK_CHECK(pvkGetSwapchainImagesKHR(s_vk.device, sc, &n, s_vk.sc_images), "vkGetSwapchainImagesKHR");

    for (u32 i = 0; i < 8; i++)
        if (s_vk.sc_done[i]) {
            pvkDestroySemaphore(s_vk.device, s_vk.sc_done[i], NULL);
            s_vk.sc_done[i] = VK_NULL_HANDLE;
        }
    VkSemaphoreCreateInfo semi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    for (u32 i = 0; i < n; i++)
        VK_CHECK(pvkCreateSemaphore(s_vk.device, &semi, NULL, &s_vk.sc_done[i]), "vkCreateSemaphore");

    s_vk.sc_extent = ext;
    s_vk.sc_count = n;
    VK_LOG("window swapchain %ux%u, %u images, format %d\n", ext.width, ext.height, n, (int)fmt.format);
    return 0;
}

static void vk_swapchain_lost(void)
{
    vk_device_idle();
    if (vk_create_swapchain()) vk_window_drop("the swapchain could not be recreated");
}

/* ---- presentation upscaler -------------------------------------------------
 * PS3RECOMP_VK_UPSCALE=1: the frame at the guest's resolution is enlarged to the swapchain's size by
 * a compute pass (vulkan/rsx_vk_upscale.comp: Catmull-Rom plus contrast-adaptive sharpening,
 * PS3RECOMP_VK_SHARP=0..1, default 0.5) into an image of that size, which is then copied 1:1 to the
 * swapchain. Off, the present is the single bilinear blit it always was. */
static struct {
    int tried, ok, on;
    float sharp;
    vk_image im;
    u32 w, h;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    VkSampler smp;
} s_up;

static int vk_up_init(void)
{
    if (s_up.tried) return s_up.ok;
    s_up.tried = 1;
    VkDescriptorSetLayoutBinding b[2] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT } };
    VkDescriptorSetLayoutCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .bindingCount = 2, .pBindings = b };
    VkPushConstantRange pr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 5 * sizeof(float) };
    VkPipelineLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                       .setLayoutCount = 1, .pSetLayouts = &s_up.dsl,
                                       .pushConstantRangeCount = 1, .pPushConstantRanges = &pr };
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
                                   { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 } };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = ps };
    VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
                                .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE };
    VkShaderModule sm = VK_NULL_HANDLE;
    if (pvkCreateDescriptorSetLayout(s_vk.device, &dci, NULL, &s_up.dsl) != VK_SUCCESS ||
        pvkCreatePipelineLayout(s_vk.device, &lci, NULL, &s_up.layout) != VK_SUCCESS ||
        pvkCreateDescriptorPool(s_vk.device, &pci, NULL, &s_up.pool) != VK_SUCCESS ||
        pvkCreateSampler(s_vk.device, &sci, NULL, &s_up.smp) != VK_SUCCESS ||
        vk_create_shader(rsx_vk_upscale_comp_spv, sizeof rsx_vk_upscale_comp_spv, &sm)) {
        VK_LOG("upscaler setup failed; presenting with the plain blit\n");
        return 0;
    }
    VkComputePipelineCreateInfo cci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" },
        .layout = s_up.layout };
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = s_up.pool, .descriptorSetCount = 1,
                                        .pSetLayouts = &s_up.dsl };
    const int ok = pvkCreateComputePipelines(s_vk.device, VK_NULL_HANDLE, 1, &cci, NULL, &s_up.pipe) == VK_SUCCESS &&
                   pvkAllocateDescriptorSets(s_vk.device, &dai, &s_up.set) == VK_SUCCESS;
    pvkDestroyShaderModule(s_vk.device, sm, NULL);
    if (!ok) VK_LOG("upscaler pipeline failed; presenting with the plain blit\n");
    s_up.ok = ok;
    return ok;
}

/* The upscaler's target at the swapchain's size (made again when it changes) and the descriptors
 * naming it and the display image. Returns 0 when the plain blit has to do. */
static int vk_up_ready(void)
{
    if (!s_up.tried) {
        const char* e = getenv("PS3RECOMP_VK_UPSCALE");
        s_up.on = e && *e && *e != '0';
        const char* sh = getenv("PS3RECOMP_VK_SHARP");
        s_up.sharp = sh && *sh ? (float)atof(sh) : 0.5f;
        if (s_up.sharp < 0.0f) s_up.sharp = 0.0f;
        if (s_up.sharp > 1.0f) s_up.sharp = 1.0f;
    }
    if (!s_up.on || !vk_up_init()) return 0;
    const u32 w = s_vk.sc_extent.width, h = s_vk.sc_extent.height;
    if (s_up.im.img && s_up.w == w && s_up.h == h) return 1;
    vk_device_idle();
    if (s_up.im.img) { vk_destroy_image(&s_up.im); memset(&s_up.im, 0, sizeof s_up.im); }
    if (vk_create_image(&s_up.im, VK_FORMAT_R8G8B8A8_UNORM, w, h,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT)) {
        vk_destroy_image(&s_up.im); memset(&s_up.im, 0, sizeof s_up.im);
        s_up.on = 0; VK_LOG("upscaler target %ux%u failed; presenting with the plain blit\n", w, h);
        return 0;
    }
    s_up.w = w; s_up.h = h;
    VkDescriptorImageInfo ii[2] = {
        { .sampler = s_up.smp, .imageView = s_vk.color.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL },
        { .imageView = s_up.im.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL } };
    VkWriteDescriptorSet wr[2] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s_up.set, .dstBinding = 0,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s_up.set, .dstBinding = 1,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &ii[1] } };
    pvkUpdateDescriptorSets(s_vk.device, 2, wr, 0, NULL);
    VK_LOG("upscaler on: %ux%u -> %ux%u, sharpness %.2f\n", s_vk.width, s_vk.height, w, h, (double)s_up.sharp);
    return 1;
}

/* Blit the offscreen target into the next swapchain image and present it. */
static void vk_present_window(void)
{
    if (!s_vk.windowed || !s_vk.swapchain) return;

    u32 idx = 0;
    VkResult r = pvkAcquireNextImageKHR(s_vk.device, s_vk.swapchain, UINT64_MAX,
                                        s_vk.sem_acquire, VK_NULL_HANDLE, &idx);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) { vk_swapchain_lost(); return; }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        VK_LOG("vkAcquireNextImageKHR failed (VkResult %d)\n", (int)r); return;
    }

    /* The present's own command buffer: free, since every caller has waited
     * for its copy (present_fence) first. */
    VkCommandBuffer cb = s_vk.present_cmd;
    if (pvkResetCommandBuffer(cb, 0) != VK_SUCCESS) { VK_LOG("vkResetCommandBuffer (window) failed\n"); return; }
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    if (pvkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) { VK_LOG("vkBeginCommandBuffer (window) failed\n"); return; }
    vk_barrier_image_on(cb, s_vk.color.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = s_vk.sc_images[idx], .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    pvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          0, 0, NULL, 0, NULL, 1, &b);
    VkImage blit_src = s_vk.color.img;
    VkImageBlit blit = {
        .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .srcOffsets = { { 0, 0, 0 }, { (int32_t)s_vk.width, (int32_t)s_vk.height, 1 } },
        .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .dstOffsets = { { 0, 0, 0 },
                        { (int32_t)s_vk.sc_extent.width, (int32_t)s_vk.sc_extent.height, 1 } },
    };
    VkFilter blit_filter = VK_FILTER_LINEAR;
    if (vk_up_ready()) {
        /* The compute pass fills s_up.im at the swapchain's size; the copy to the swapchain is then 1:1. */
        vk_barrier_image_on(cb, s_up.im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
        const float pc[5] = { (float)s_vk.width, (float)s_vk.height,
                              (float)s_up.w, (float)s_up.h, s_up.sharp };
        pvkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s_up.pipe);
        pvkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s_up.layout, 0, 1, &s_up.set, 0, NULL);
        pvkCmdPushConstants(cb, s_up.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
        pvkCmdDispatch(cb, (s_up.w + 7u) / 8u, (s_up.h + 7u) / 8u, 1);
        VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                               .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT };
        pvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 1, &mb, 0, NULL, 0, NULL);
        blit_src = s_up.im.img;
        blit.srcOffsets[1] = (VkOffset3D){ (int32_t)s_up.w, (int32_t)s_up.h, 1 };
        blit_filter = VK_FILTER_NEAREST;
    }
    pvkCmdBlitImage(cb, blit_src, VK_IMAGE_LAYOUT_GENERAL,
                    s_vk.sc_images[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &blit, blit_filter);
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = 0;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    pvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                          0, 0, NULL, 0, NULL, 1, &b);
    if (pvkEndCommandBuffer(cb) != VK_SUCCESS) { VK_LOG("vkEndCommandBuffer failed\n"); return; }

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1, .pWaitSemaphores = &s_vk.sem_acquire,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1, .pCommandBuffers = &cb,
        .signalSemaphoreCount = 1, .pSignalSemaphores = &s_vk.sc_done[idx],
    };
    pvkResetFences(s_vk.device, 1, &s_vk.window_fence);
    if (vk_queue_submit(&si, s_vk.window_fence) != VK_SUCCESS) {
        VK_LOG("vkQueueSubmit (window) failed\n"); return;
    }
    pvkWaitForFences(s_vk.device, 1, &s_vk.window_fence, VK_TRUE, UINT64_MAX);

    VkPresentInfoKHR pi = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1, .pWaitSemaphores = &s_vk.sc_done[idx],
        .swapchainCount = 1, .pSwapchains = &s_vk.swapchain, .pImageIndices = &idx,
    };
    r = vk_queue_present(&pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) vk_swapchain_lost();
    else if (r != VK_SUCCESS) VK_LOG("vkQueuePresentKHR failed (VkResult %d)\n", (int)r);
}

static void vk_dump_ppm(const unsigned char* rgba)
{
    /* A path with a %u (e.g. frame_%05u.ppm) gets the present count, so a run
     * leaves a sequence instead of overwriting one file. */
    char path[1024];
    if (strstr(s_vk.dump_path, "%u") || strstr(s_vk.dump_path, "%05u"))
        snprintf(path, sizeof path, s_vk.dump_path, (unsigned)s_vk.presented);
    else
        snprintf(path, sizeof path, "%s", s_vk.dump_path);
    FILE* f = fopen(path, "wb");
    if (!f) { VK_LOG("cannot open %s for the frame dump\n", path); return; }
    fprintf(f, "P6\n%u %u\n255\n", s_vk.width, s_vk.height);
    for (size_t i = 0; i < (size_t)s_vk.width * s_vk.height; i++)
        fwrite(rgba + i * 4, 1, 3, f);
    fclose(f);
}

/* ---------------------------------------------------------------------------
 * Present
 *
 * In two halves. The first records, on the present's own command buffer,
 * whatever puts the frame into the offscreen target (the engine's blit of
 * the flipped surface) and the copy of that target into `readback`, and
 * submits it without waiting. The second waits for it, reads the centre
 * pixel and the frame dump back, and blits the target into the window.
 *
 * The first half reads the frame, so it must not run while the walker is
 * recording; the second touches only the target, `readback` and the
 * present's own objects, none of which the engine path draws into, so a host
 * can run it on a thread of its own beside the walker
 * (rsx_vulkan_backend_present_begin / _finish). On the fixed-function path
 * the walker draws into the target itself, so there the second half always
 * follows the first at once.
 * -------------------------------------------------------------------------*/

/* Wait until the GPU is done with a present's copy. Safe from any thread:
 * present_fence is reset only when a present starts, which never runs
 * alongside the walker. */
static void vk_present_wait_gpu(void)
{
    if (__atomic_load_n(&s_vk.present_inflight, __ATOMIC_ACQUIRE) && s_vk.present_fence)
        pvkWaitForFences(s_vk.device, 1, &s_vk.present_fence, VK_TRUE, UINT64_MAX);
}

/* present_mu held. */
static void vk_present_complete_locked(void)
{
    if (!s_vk.present_inflight) return;
    pvkWaitForFences(s_vk.device, 1, &s_vk.present_fence, VK_TRUE, UINT64_MAX);
    __atomic_store_n(&s_vk.present_inflight, 0, __ATOMIC_RELEASE);

    vk_invalidate(&s_vk.readback);
    const unsigned char* px = (const unsigned char*)s_vk.readback.ptr;
    size_t c = ((size_t)(s_vk.height / 2) * s_vk.width + s_vk.width / 2) * 4;
    s_vk.last_center = ((u32)px[c] << 16) | ((u32)px[c + 1] << 8) | (u32)px[c + 2];
    s_vk.presented++;
    if (s_vk.dump_path && (s_vk.presented % s_vk.dump_every) == 0) vk_dump_ppm(px);
    vk_present_window();
}

/* present_mu held: a previous present finished, the present's command
 * buffer begun and ordered after everything submitted so far. */
static VkCommandBuffer vk_present_record_begin(void)
{
    vk_present_complete_locked();
    VkCommandBuffer cb = s_vk.present_cmd;
    if (pvkResetCommandBuffer(cb, 0) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    if (pvkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) return VK_NULL_HANDLE;
    vk_cmd_after_everything(cb);
    return cb;
}

/* present_mu held: copy the target out, submit, and finish at once unless a
 * host asked for the halves separately on the engine path. */
static void vk_present_record_end(VkCommandBuffer cb)
{
    vk_barrier_image_on(cb, s_vk.color.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    VkBufferImageCopy region = {
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .imageExtent = { s_vk.width, s_vk.height, 1 },
    };
    pvkCmdCopyImageToBuffer(cb, s_vk.color.img, VK_IMAGE_LAYOUT_GENERAL,
                            s_vk.readback.buf, 1, &region);
    VkBufferMemoryBarrier host = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = s_vk.readback.buf, .offset = 0, .size = VK_WHOLE_SIZE,
    };
    pvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                          0, 0, NULL, 1, &host, 0, NULL);
    if (pvkEndCommandBuffer(cb) != VK_SUCCESS) { VK_LOG("vkEndCommandBuffer (present) failed\n"); return; }
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1, .pCommandBuffers = &cb };
    pvkResetFences(s_vk.device, 1, &s_vk.present_fence);
    if (vk_queue_submit(&si, s_vk.present_fence) != VK_SUCCESS) {
        VK_LOG("vkQueueSubmit (present) failed\n"); return;
    }
    __atomic_store_n(&s_vk.present_inflight, 1, __ATOMIC_RELEASE);
    s_vk.present_order = 1;
    if (!(s_vk.present_split && s_vk.eng_active)) vk_present_complete_locked();
}

static void vk_cb_present(void* ud, u32 buffer_id)
{
    (void)ud; (void)buffer_id;
    if (!s_vk.device) return;
    pthread_mutex_lock(&s_vk.present_mu);
    VkCommandBuffer cb = vk_present_record_begin();
    if (cb) vk_present_record_end(cb);
    pthread_mutex_unlock(&s_vk.present_mu);
    s_g.last_draws = s_g.draws;
    s_g.draws = 0;
}

/* ---------------------------------------------------------------------------
 * Register-file draw engine backend (rsx_draw_engine.h)
 *
 * The path the Metal backend takes: the shared engine walks the register
 * file, owns surfaces, texture caching, vertex compaction and pipeline keys,
 * and drives this backend through rsx_draw_backend. Selected with
 * PS3RECOMP_RSX_ENGINE=dispatch; only one of the two paths is registered at a
 * time, since the FIFO walker feeds both and would record every draw twice.
 *
 * Every operation is still one synchronous submission, so submit_and_wait has
 * nothing left to do by the time the engine calls it -- which is exactly the
 * contract it asks for (staging is free to reuse on return).
 *
 * Stage E1: colour and depth targets, clears, present and readback.
 * Pipelines, draws and textures are the next stages; until then
 * pipeline_create reports 0, which the engine caches as "cannot build".
 * -------------------------------------------------------------------------*/
#define VK_ENG_MAX_OBJ 4096          /* the engine's texture cache is 3072 */

enum { VK_ENG_FREE = 0, VK_ENG_COLOR, VK_ENG_DEPTH, VK_ENG_TEXTURE, VK_ENG_VIEW };

typedef struct vk_eng_obj {
    u8       kind;
    VkFormat fmt;
    u32      bpp;                    /* bytes per texel, 0 for depth     */
    u32      w, h;
    vk_image im;                     /* a VIEW owns only im.view            */
    u32      block;                  /* bytes per 4x4 block, 0 = not BC     */
    u32      surface, remap, rsx_fmt;/* VIEW: what it is a view of          */
    u8       cube;                   /* TEXTURE: six layers, a CUBE view    */
    u8       snapshot;               /* COLOR: a feedback copy, never a target */
    u32      mips;                   /* TEXTURE: levels the image was created with */
    u32      depth3d;                /* TEXTURE with cube == 2: slices of the 3D image */
} vk_eng_obj;

static vk_eng_obj s_eobj[VK_ENG_MAX_OBJ];

/* Defined with the E2 pipeline state further down. */
static void vk_eng_drop_framebuffers(void);
static void vk_eng2_shutdown(void* user);

static struct {
    u32 rt[RSX_BE_MAX_COLOR_TARGETS], nrt, depth;
} s_eng_bound;

static VkFormat vk_eng_format(rsx_be_format f, u32* bpp)
{
    switch (f) {
    case RSX_BE_FMT_R8:            *bpp = 1;  return VK_FORMAT_R8_UNORM;
    case RSX_BE_FMT_R8G8:          *bpp = 2;  return VK_FORMAT_R8G8_UNORM;
    case RSX_BE_FMT_R8G8B8A8:      *bpp = 4;  return VK_FORMAT_R8G8B8A8_UNORM;
    case RSX_BE_FMT_R16:           *bpp = 2;  return VK_FORMAT_R16_UNORM;
    case RSX_BE_FMT_R16G16:        *bpp = 4;  return VK_FORMAT_R16G16_UNORM;
    case RSX_BE_FMT_R16G16F:       *bpp = 4;  return VK_FORMAT_R16G16_SFLOAT;
    case RSX_BE_FMT_R16G16B16A16F: *bpp = 8;  return VK_FORMAT_R16G16B16A16_SFLOAT;
    case RSX_BE_FMT_R32F:          *bpp = 4;  return VK_FORMAT_R32_SFLOAT;
    case RSX_BE_FMT_R32G32B32A32F: *bpp = 16; return VK_FORMAT_R32G32B32A32_SFLOAT;
    case RSX_BE_FMT_BC1:           *bpp = 8;  return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;  /* per block */
    case RSX_BE_FMT_BC2:           *bpp = 16; return VK_FORMAT_BC2_UNORM_BLOCK;
    case RSX_BE_FMT_BC3:           *bpp = 16; return VK_FORMAT_BC3_UNORM_BLOCK;
    default:                       *bpp = 0;  return VK_FORMAT_UNDEFINED;
    }
}

/* Colour formats the E2 passes render into: RGBA8 for ordinary and display
 * targets, RGBA16F for SET_SURFACE_FORMAT 0xB HDR targets, R32F for 0xD
 * (F_X32: GoW:A renders 1280x720 single-float targets). All three are required
 * colour attachment formats in Vulkan 1.0; blending on R32F is optional, and the
 * RSX does not blend FP32 targets either, so pipelines for it never blend. */
#define VK_ENG_RT_FORMATS 3
static int vk_eng_rt_index(VkFormat f)
{
    switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM:      return 0;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return 1;
    case VK_FORMAT_R32_SFLOAT:          return 2;
    default:                            return -1;
    }
}

/* The largest image side the engine creates: the device's own limit once a
 * device is picked, the 4096 Vulkan guarantees before. */
static u32 vk_max_dim(void) { return s_vk.max_dim ? s_vk.max_dim : 4096u; }

static u32 vk_eng_alloc(void)
{
    for (u32 i = 1; i < VK_ENG_MAX_OBJ; i++)
        if (s_eobj[i].kind == VK_ENG_FREE) return i;
    VK_LOG("engine: object table full (%d)\n", VK_ENG_MAX_OBJ);
    return 0;
}

static vk_eng_obj* vk_eng_get(u32 handle, int kind)
{
    if (!handle || handle >= VK_ENG_MAX_OBJ) return NULL;
    vk_eng_obj* o = &s_eobj[handle];
    return (o->kind == kind) ? o : NULL;
}

/* Host staging kept across readbacks and unbatched uploads. One buffer
 * allocated, mapped and freed per call is three driver round trips, and on
 * Windows those took from tens of milliseconds to seconds each under load
 * (VK_SLOW_MS). Each grows to the largest request and is reused after its
 * submission completes; every user waits for that before returning. */
static vk_buffer s_rb_stage, s_up_stage;
static vk_buffer* vk_stage_get(vk_buffer* b, VkDeviceSize size, VkBufferUsageFlags usage)
{
    if (b->buf && b->size >= size) return b;
    if (b->buf) vk_destroy_buffer(b);
    VkDeviceSize cap = (VkDeviceSize)1 << 20;
    while (cap < size) cap *= 2u;
    if (vk_create_host_buffer(b, cap, usage)) { vk_destroy_buffer(b); return NULL; }
    return b;
}

/* Copy `rows` rows of `w` texels from host memory into a w x h rectangle at
 * (x, y) of one level / layer of an image that is in GENERAL. */
static int vk_eng_upload_region_z(VkImage img, u32 layer, u32 mip, u32 x, u32 y, u32 z,
                                  u32 w, u32 h, size_t tight, u32 rows,
                                  const void* src, u32 row_bytes);
static int vk_eng_upload_region(VkImage img, u32 layer, u32 mip, u32 x, u32 y,
                                u32 w, u32 h, size_t tight, u32 rows,
                                const void* src, u32 row_bytes)
{
    return vk_eng_upload_region_z(img, layer, mip, x, y, 0, w, h, tight, rows, src, row_bytes);
}
/* z: the slice of a 3D image (vk_eng_texture_upload_3d); 0 otherwise. */
static int vk_eng_upload_region_z(VkImage img, u32 layer, u32 mip, u32 x, u32 y, u32 z,
                                  u32 w, u32 h, size_t tight, u32 rows,
                                  const void* src, u32 row_bytes)
{
    if (tight > row_bytes) tight = row_bytes;
    {   /* VK_DRAW_BATCH: staged in a ring and recorded into the open batch. */
        VkBufferImageCopy region = {
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1 },
            .imageOffset = { (int32_t)x, (int32_t)y, (int32_t)z },
            .imageExtent = { w, h, 1 },
        };
        const int b = vk_eb_upload(img, &region, src, tight, rows, row_bytes);
        if (b != -2) return b;
    }
    vk_buffer* const st = vk_stage_get(&s_up_stage, (VkDeviceSize)tight * rows, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    if (!st || !st->coherent) return -1;
    for (u32 y = 0; y < rows; y++)
        memcpy((u8*)st->ptr + (size_t)y * tight, (const u8*)src + (size_t)y * row_bytes, tight);
    int rc = vk_begin();
    if (!rc) {
        vk_barrier_image(img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
        VkBufferImageCopy region = {
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1 },
            .imageOffset = { (int32_t)x, (int32_t)y, (int32_t)z },
            .imageExtent = { w, h, 1 },
        };
        pvkCmdCopyBufferToImage(s_vk.cmd, st->buf, img, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        rc = vk_submit_and_wait();
    }
    return rc;
}

/* The same into the whole of a level, from its origin. */
static int vk_eng_upload_rows(VkImage img, u32 layer, u32 mip, u32 w, u32 h,
                              size_t tight, u32 rows, const void* src, u32 row_bytes)
{
    return vk_eng_upload_region(img, layer, mip, 0, 0, w, h, tight, rows, src, row_bytes);
}

static int vk_eng_upload(VkImage img, u32 layer, u32 mip, u32 w, u32 h, u32 bpp,
                         const void* src, u32 row_bytes)
{
    return vk_eng_upload_rows(img, layer, mip, w, h, (size_t)w * bpp, h, src, row_bytes);
}

/* NV4097 TEXTURE_CONTROL1 crossbar as a Vulkan component mapping: the shared
 * rsx_texture_component_remap() gives, in A,R,G,B order, which uploaded
 * channel (0..3 = R,G,B,A) or constant each output takes -- the table the
 * Metal backend's swizzle_sel reads. */
static VkComponentSwizzle vk_remap_sel(u8 sel)
{
    switch (sel) {
    case 0: return VK_COMPONENT_SWIZZLE_R;
    case 1: return VK_COMPONENT_SWIZZLE_G;
    case 2: return VK_COMPONENT_SWIZZLE_B;
    case 3: return VK_COMPONENT_SWIZZLE_A;
    case RSX_REMAP_ONE: return VK_COMPONENT_SWIZZLE_ONE;
    default: return VK_COMPONENT_SWIZZLE_ZERO;
    }
}
static VkComponentMapping vk_remap(u32 remap, u32 rsx_fmt)
{
    u8 sel[4];
    rsx_texture_component_remap(remap, rsx_fmt & 0x9Fu, sel);
    VkComponentMapping m = { vk_remap_sel(sel[1]), vk_remap_sel(sel[2]),
                             vk_remap_sel(sel[3]), vk_remap_sel(sel[0]) };
    return m;
}
static int vk_eng_make_view(VkImage img, VkFormat fmt, u32 levels, VkComponentMapping map,
                            int cube, VkImageView* out)
{
    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
        .viewType = cube == 2 ? VK_IMAGE_VIEW_TYPE_3D : cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D,
        .format = fmt, .components = map,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels ? levels : 1, 0, cube == 1 ? 6u : 1u },
    };
    return pvkCreateImageView(s_vk.device, &vci, NULL, out) == VK_SUCCESS ? 0 : -1;
}

static int vk_eng_init(void* user, u32 width, u32 height)
{
    (void)user; (void)width; (void)height;
    memset(s_eobj, 0, sizeof s_eobj);
    memset(&s_eng_bound, 0, sizeof s_eng_bound);
    return s_vk.device ? 0 : -1;
}

static void vk_eng_snapshot_recycle(vk_eng_obj* o);

static void vk_eng_release(void* user, u32 handle)
{
    (void)user;
    if (!handle || handle >= VK_ENG_MAX_OBJ || s_eobj[handle].kind == VK_ENG_FREE) return;
    if (s_eobj[handle].kind == VK_ENG_COLOR)
        for (u32 i = 1; i < VK_ENG_MAX_OBJ; i++)
            if (s_eobj[i].kind == VK_ENG_VIEW && s_eobj[i].surface == handle) {
                vk_destroy_image(&s_eobj[i].im);
                memset(&s_eobj[i], 0, sizeof s_eobj[i]);
            }
    if (s_eobj[handle].kind == VK_ENG_DEPTH && s_eobj[handle].surface) {
        const u32 snap = s_eobj[handle].surface;
        s_eobj[handle].surface = 0;
        if (vk_eng_get(snap, VK_ENG_TEXTURE)) vk_eng_release(user, snap);
    }
    if (s_eobj[handle].kind == VK_ENG_COLOR && s_eobj[handle].snapshot) {
        /* Never an attachment, so no framebuffer names it; the image goes
         * back to the pool for the next feedback draw. */
        vk_eng_snapshot_recycle(&s_eobj[handle]);
        memset(&s_eobj[handle], 0, sizeof s_eobj[handle]);
        return;
    }
    if (s_eobj[handle].kind == VK_ENG_COLOR || s_eobj[handle].kind == VK_ENG_DEPTH)
        vk_eng_drop_framebuffers();
    /* A texture is only ever sampled or copied by the engine's own batches:
     * destroyed once they have run (vk_grave_push), not after a full wait. */
    if (s_eobj[handle].kind == VK_ENG_TEXTURE &&
        vk_grave_push(s_eobj[handle].im.img, s_eobj[handle].im.view, s_eobj[handle].im.mem)) {
        memset(&s_eobj[handle], 0, sizeof s_eobj[handle]);
        return;
    }
    vk_destroy_image(&s_eobj[handle].im);      /* idle: submissions are waited on */
    memset(&s_eobj[handle], 0, sizeof s_eobj[handle]);
}

static void vk_eng_snapshot_pool_drain(void);

static void vk_eng_shutdown(void* user)
{
    vk_eng2_shutdown(user);
    for (u32 i = 1; i < VK_ENG_MAX_OBJ; i++) vk_eng_release(user, i);
    vk_eng_idle();                 /* and the textures waiting for their batches go */
    vk_eng_snapshot_pool_drain();
    vk_destroy_buffer(&s_rb_stage);
    vk_destroy_buffer(&s_up_stage);
}

static void vk_eng_submit_and_wait(void* user, u32 reason)
{
    (void)user; (void)reason;
    vk_eng_idle();                 /* the engine reads results next */
}

static u32 vk_eng_color_target_create(void* user, rsx_be_format fmt, u32 w, u32 h,
                                      const void* seed, u32 seed_row_bytes)
{
    (void)user;
    u32 bpp;
    const VkFormat vf = vk_eng_format(fmt, &bpp);
    /* Only formats the E2 passes are built for (vk_eng_rt_index). */
    if (vk_eng_rt_index(vf) < 0 || !w || !h || w > vk_max_dim() || h > vk_max_dim()) {
        VK_LOG("engine: colour target format %d %ux%u not supported yet\n", (int)fmt, w, h);
        return 0;
    }
    const u32 hd = vk_eng_alloc();
    if (!hd) return 0;
    vk_eng_obj* o = &s_eobj[hd];
    if (vk_create_image(&o->im, vf, w, h,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT)) {
        vk_destroy_image(&o->im); return 0;
    }
    o->kind = VK_ENG_COLOR; o->fmt = vf; o->bpp = bpp; o->w = w; o->h = h;
    /* UNDEFINED -> GENERAL, then the guest's own bytes when the engine could
     * resolve them, else transparent black so the contents are defined. */
    if (vk_begin()) { vk_eng_release(user, hd); return 0; }
    vk_barrier_image(o->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
    VkClearColorValue zero = { .float32 = { 0.0f, 0.0f, 0.0f, 0.0f } };
    VkImageSubresourceRange r = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    pvkCmdClearColorImage(s_vk.cmd, o->im.img, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &r);
    if (vk_submit_and_wait()) { vk_eng_release(user, hd); return 0; }
    if (seed && seed_row_bytes)
        vk_eng_upload(o->im.img, 0, 0, w, h, bpp, seed, seed_row_bytes);
    return hd;
}

static u32 vk_eng_depth_target_create(void* user, u32 w, u32 h)
{
    (void)user;
    if (!w || !h || w > vk_max_dim() || h > vk_max_dim()) return 0;
    const u32 hd = vk_eng_alloc();
    if (!hd) return 0;
    vk_eng_obj* o = &s_eobj[hd];
    if (vk_create_image(&o->im, s_vk.depth_format, w, h,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        s_vk.depth_aspect)) {
        vk_destroy_image(&o->im); return 0;
    }
    o->kind = VK_ENG_DEPTH; o->fmt = s_vk.depth_format; o->w = w; o->h = h;
    /* Far plane, for the reason the display depth starts there. */
    if (vk_begin()) { vk_eng_release(user, hd); return 0; }
    vk_barrier_image(o->im.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_UNDEFINED);
    VkClearDepthStencilValue far_plane = { 1.0f, 0 };
    VkImageSubresourceRange r = { s_vk.depth_aspect, 0, 1, 0, 1 };
    pvkCmdClearDepthStencilImage(s_vk.cmd, o->im.img, VK_IMAGE_LAYOUT_GENERAL, &far_plane, 1, &r);
    if (vk_submit_and_wait()) { vk_eng_release(user, hd); return 0; }
    return hd;
}

static void vk_eng_bind_targets(void* user, const u32* surfaces, u32 count, u32 depth)
{
    (void)user;
    if (count > RSX_BE_MAX_COLOR_TARGETS) count = RSX_BE_MAX_COLOR_TARGETS;
    for (u32 i = 0; i < RSX_BE_MAX_COLOR_TARGETS; i++)
        s_eng_bound.rt[i] = (i < count && surfaces) ? surfaces[i] : 0;
    s_eng_bound.nrt = count;
    s_eng_bound.depth = depth;
}

static void vk_eng_clear_color(void* user, u32 surface, const float rgba[4])
{
    (void)user;
    /* The debug hook reports the last colour asked for, as Metal's does. */
    s_vk.clear_argb = ((u32)(rgba[3] * 255.0f + 0.5f) << 24) |
                      ((u32)(rgba[0] * 255.0f + 0.5f) << 16) |
                      ((u32)(rgba[1] * 255.0f + 0.5f) <<  8) |
                       (u32)(rgba[2] * 255.0f + 0.5f);
    vk_eng_obj* o = vk_eng_get(surface, VK_ENG_COLOR);
    if (!o) return;
    const int batched = vk_eb_active();
    if (batched ? (!s_eb_pending && vk_begin()) : vk_begin()) return;
    VkClearColorValue v;
    memcpy(v.float32, rgba, sizeof v.float32);
    VkImageSubresourceRange r = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vk_barrier_image(o->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    pvkCmdClearColorImage(s_vk.cmd, o->im.img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
    if (batched) vk_eb_recorded();
    else vk_submit_and_wait();
}

static void vk_eng_clear_depth_stencil(void* user, u32 depth, u32 flags,
                                       float depth_value, u8 stencil)
{
    (void)user;
    vk_eng_obj* o = vk_eng_get(depth, VK_ENG_DEPTH);
    VkImageAspectFlags aspect = 0;
    if (flags & RSX_BE_CLEAR_DEPTH) aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (flags & RSX_BE_CLEAR_STENCIL) aspect |= s_vk.depth_aspect & VK_IMAGE_ASPECT_STENCIL_BIT;
    if (!o || !aspect) return;
    const int batched = vk_eb_active();
    if (batched ? (!s_eb_pending && vk_begin()) : vk_begin()) return;
    VkClearDepthStencilValue d = { depth_value, stencil };
    VkImageSubresourceRange r = { aspect, 0, 1, 0, 1 };
    vk_barrier_image(o->im.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
    pvkCmdClearDepthStencilImage(s_vk.cmd, o->im.img, VK_IMAGE_LAYOUT_GENERAL, &d, 1, &r);
    if (batched) vk_eb_recorded();
    else vk_submit_and_wait();
}

/* The named surface becomes the frame: blit it onto the display image, then
 * the existing present does the readback, the PPM dump and the window. */
static void vk_eng_dump_surfaces(u32 present_no, const char* dir);
static void vk_eng_present(void* user, u32 surface)
{
    (void)user;
    if (!s_vk.device) return;
    if (s_eb_pending) vk_eng_flush();   /* the present records on its own buffer */
    { static int s_at = -2; static char s_dir[512];
      if (s_at == -2) {
          const char* e = getenv("PS3RECOMP_VK_DUMP_SURFACES");
          s_at = -1;
          if (e) { const char* c = strchr(e, ',');
                   if (c) { s_at = atoi(e); snprintf(s_dir, sizeof s_dir, "%s", c + 1); } }
      }
      if (s_at >= 0 && s_vk.presented + 1 == (u32)s_at) {
          VK_LOG("surface dump at present %u (presenting handle %u)\n", (unsigned)s_at, surface);
          vk_eng_dump_surfaces((u32)s_at, s_dir);
      } }
    vk_eng_obj* o = vk_eng_get(surface, VK_ENG_COLOR);
    pthread_mutex_lock(&s_vk.present_mu);
    VkCommandBuffer cb = vk_present_record_begin();
    if (cb) {
        if (o) {
            vk_barrier_image_on(cb, o->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
            vk_barrier_image_on(cb, s_vk.color.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
            VkImageBlit blit = {
                .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                .srcOffsets = { { 0, 0, 0 }, { (int32_t)o->w, (int32_t)o->h, 1 } },
                .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                .dstOffsets = { { 0, 0, 0 }, { (int32_t)s_vk.width, (int32_t)s_vk.height, 1 } },
            };
            pvkCmdBlitImage(cb, o->im.img, VK_IMAGE_LAYOUT_GENERAL,
                            s_vk.color.img, VK_IMAGE_LAYOUT_GENERAL, 1, &blit,
                            (o->w == s_vk.width && o->h == s_vk.height) ? VK_FILTER_NEAREST
                                                                        : VK_FILTER_LINEAR);
        }
        vk_present_record_end(cb);
    }
    pthread_mutex_unlock(&s_vk.present_mu);
    s_g.last_draws = s_g.draws;
    s_g.draws = 0;
}

/* Pixels a 2D transfer put into guest memory, pushed into the colour target
 * that rectangle belongs to. */
static void vk_eng_color_target_write(void* user, u32 surface, u32 x, u32 y,
                                      u32 w, u32 h, const void* rows, u32 row_bytes)
{
    (void)user;
    vk_eng_obj* o = vk_eng_get(surface, VK_ENG_COLOR);
    if (!o || o->fmt != VK_FORMAT_R8G8B8A8_UNORM) return;   /* rows are R,G,B,A */
    if (!rows || !w || !h || x + w > o->w || y + h > o->h) return;
    vk_eng_upload_region(o->im.img, 0, 0, x, y, w, h, (size_t)w * o->bpp, h, rows, row_bytes);
}

static void vk_eng_readback(void* user, u32 surface, u32 x, u32 y, u32 w, u32 h,
                            void* out, u32 out_pitch)
{
    (void)user;
    vk_eng_obj* o = vk_eng_get(surface, VK_ENG_COLOR);
    if (!o || !out || !out_pitch || !w || !h || x + w > o->w || y + h > o->h) return;
    const size_t tight = (size_t)w * o->bpp;
    vk_buffer* const st = vk_stage_get(&s_rb_stage, (VkDeviceSize)tight * h, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!st) return;
    if (!vk_begin()) {
        vk_barrier_image(o->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
        VkBufferImageCopy region = {
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .imageOffset = { (int32_t)x, (int32_t)y, 0 }, .imageExtent = { w, h, 1 },
        };
        pvkCmdCopyImageToBuffer(s_vk.cmd, o->im.img, VK_IMAGE_LAYOUT_GENERAL, st->buf, 1, &region);
        VkBufferMemoryBarrier hb = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = st->buf, .offset = 0, .size = VK_WHOLE_SIZE,
        };
        pvkCmdPipelineBarrier(s_vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                              0, 0, NULL, 1, &hb, 0, NULL);
        if (!vk_submit_and_wait()) {
            vk_invalidate(st);
            for (u32 row = 0; row < h; row++)
                memcpy((u8*)out + (size_t)row * out_pitch, (const u8*)st->ptr + (size_t)row * tight, tight);
        }
    }
}

/* PS3RECOMP_VK_DUMP_SURFACES=<present#>,<dir>: at that present, write every
 * engine colour target to <dir>/surf_<present>_h<handle>_<w>x<h>_<fmt>.ppm and
 * log its value range -- which pass of a multi-pass frame still holds the
 * image. RGBA16F is clamped to [0,1]; R32F is scaled by its own maximum. */
static float vk_half_to_float(u16 h)
{
    const u32 s = (u32)(h >> 15) << 31, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    u32 bits;
    if (e == 0) {
        if (!m) bits = s;
        else { float f = (float)m / 16777216.0f; if (s) f = -f; return f; }   /* denormal */
    } else if (e == 31) bits = s | 0x7F800000u | (m << 13);
    else bits = s | ((e + 112u) << 23) | (m << 13);
    float f; memcpy(&f, &bits, 4); return f;
}
static void vk_eng_dump_surfaces(u32 present_no, const char* dir)
{
    for (u32 hd = 1; hd < VK_ENG_MAX_OBJ; hd++) {
        vk_eng_obj* o = &s_eobj[hd];
        if (o->kind != VK_ENG_COLOR || !o->w || !o->h || !o->bpp) continue;
        const u32 w = o->w, h = o->h, bpp = o->bpp;
        u8* raw = (u8*)malloc((size_t)w * h * bpp);
        u8* rgb = (u8*)malloc((size_t)w * h * 3);
        if (!raw || !rgb) { free(raw); free(rgb); continue; }
        memset(raw, 0, (size_t)w * h * bpp);
        vk_eng_readback(NULL, hd, 0, 0, w, h, raw, w * bpp);
        float mx = 0.0f; size_t nz = 0;
        for (size_t i = 0; i < (size_t)w * h; i++) {
            float c[3] = { 0, 0, 0 };
            if (o->fmt == VK_FORMAT_R8G8B8A8_UNORM) {
                for (int k = 0; k < 3; k++) c[k] = raw[i * 4 + k] / 255.0f;
            } else if (o->fmt == VK_FORMAT_R16G16B16A16_SFLOAT) {
                for (int k = 0; k < 3; k++) { u16 v; memcpy(&v, raw + i * 8 + k * 2, 2); c[k] = vk_half_to_float(v); }
            } else if (o->fmt == VK_FORMAT_R32_SFLOAT) {
                float v; memcpy(&v, raw + i * 4, 4); c[0] = c[1] = c[2] = v;
            }
            for (int k = 0; k < 3; k++) {
                if (c[k] != 0.0f) nz++;
                if (c[k] > mx) mx = c[k];
            }
        }
        const float scale = (o->fmt == VK_FORMAT_R32_SFLOAT && mx > 0.0f) ? 1.0f / mx : 1.0f;
        for (size_t i = 0; i < (size_t)w * h; i++) {
            float c[3] = { 0, 0, 0 };
            if (o->fmt == VK_FORMAT_R8G8B8A8_UNORM) {
                for (int k = 0; k < 3; k++) c[k] = raw[i * 4 + k] / 255.0f;
            } else if (o->fmt == VK_FORMAT_R16G16B16A16_SFLOAT) {
                for (int k = 0; k < 3; k++) { u16 v; memcpy(&v, raw + i * 8 + k * 2, 2); c[k] = vk_half_to_float(v); }
            } else if (o->fmt == VK_FORMAT_R32_SFLOAT) {
                float v; memcpy(&v, raw + i * 4, 4); c[0] = c[1] = c[2] = v;
            }
            for (int k = 0; k < 3; k++) {
                float v = c[k] * scale; v = v < 0 ? 0 : (v > 1 ? 1 : v);
                rgb[i * 3 + k] = (u8)(v * 255.0f + 0.5f);
            }
        }
        char path[1024];
        snprintf(path, sizeof path, "%s/surf_%05u_h%02u_%ux%u_f%d.ppm", dir, present_no, hd, w, h, (int)o->fmt);
        FILE* f = fopen(path, "wb");
        if (f) { fprintf(f, "P6\n%u %u\n255\n", w, h); fwrite(rgb, 1, (size_t)w * h * 3, f); fclose(f); }
        VK_LOG("surface dump h%u %ux%u fmt %d: max %.4g, nonzero components %zu of %zu -> %s\n",
               hd, w, h, (int)o->fmt, mx, nz, (size_t)w * h * 3, path);
        free(raw); free(rgb);
    }
}

/* ---- E3: textures ---------------------------------------------------------
 * The engine decodes every level to host rows (rsx_texture_decode) and hands
 * them over one upload each; the crossbar is a property of the view, so it
 * becomes the image view's component mapping. A cube map (faces == 6) is a
 * six-layer cube-compatible image, each face uploaded into its layer, sampled
 * through a CUBE view; the pipelines that sample one use the split set layout
 * (vk_eng_split_layout). */
static u32 vk_eng_texture_create(void* user, rsx_be_format fmt, u32 w, u32 h,
                                 u32 mips, u32 faces, u32 remap, u32 rsx_fmt)
{
    (void)user;
    u32 bpp;
    const VkFormat vf = vk_eng_format(fmt, &bpp);
    const int bc = (fmt == RSX_BE_FMT_BC1 || fmt == RSX_BE_FMT_BC2 || fmt == RSX_BE_FMT_BC3);
    if (vf == VK_FORMAT_UNDEFINED || !w || !h || w > vk_max_dim() || h > vk_max_dim() ||
        (bc && !s_vk.bc_ok))
        return 0;
    const int cube = (faces == 6);
    if ((faces != 1 && !cube) || (cube && w != h)) return 0;   /* cube faces are square */
    if (!mips) mips = 1;
    const u32 hd = vk_eng_alloc();
    if (!hd) return 0;
    vk_eng_obj* o = &s_eobj[hd];
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0, .imageType = VK_IMAGE_TYPE_2D,
        .format = vf, .extent = { w, h, 1 }, .mipLevels = mips, .arrayLayers = cube ? 6u : 1u,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        /* TRANSFER_SRC too: vk_eng_image_copy (the MLAA strips) copies out of a texture. */
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
    VkMemoryRequirements req;
    u32 type;
    if (pvkCreateImage(s_vk.device, &ici, NULL, &o->im.img) != VK_SUCCESS) goto fail;
    pvkGetImageMemoryRequirements(s_vk.device, o->im.img, &req);
    if (vk_find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &type) &&
        vk_find_memory_type(req.memoryTypeBits, 0, &type)) goto fail;
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = req.size, .memoryTypeIndex = type };
    if (pvkAllocateMemory(s_vk.device, &mai, NULL, &o->im.mem) != VK_SUCCESS ||
        pvkBindImageMemory(s_vk.device, o->im.img, o->im.mem, 0) != VK_SUCCESS ||
        vk_eng_make_view(o->im.img, vf, mips, vk_remap(remap, rsx_fmt), cube, &o->im.view))
        goto fail;
    o->kind = VK_ENG_TEXTURE; o->fmt = vf; o->w = w; o->h = h; o->cube = (u8)cube; o->mips = mips;
    o->bpp = bc ? 0 : bpp; o->block = bc ? bpp : 0;
    /* UNDEFINED -> GENERAL for every level and face before the uploads arrive.
     * With draw batching it is one more operation of the open batch, ahead of
     * the uploads that follow it there (vk_eb_upload), like the clears: as a
     * synchronous submit it waited for every batch in flight (one in-order
     * queue) once per new texture -- ~6 a frame at ~0.9 ms each while a GoW:A
     * scene streams its textures in. VK_TEX_CREATE_SYNC=1 keeps the wait. */
    {   static int s_sync = -1;
        if (s_sync < 0) { const char* e = getenv("VK_TEX_CREATE_SYNC"); s_sync = (e && *e == '1') ? 1 : 0; }
        const int batched = vk_eb_active() && !s_sync;
        if (batched ? (!s_eb_pending && vk_begin()) : vk_begin()) goto fail;
        vk_barrier_image(o->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
        if (batched) vk_eb_recorded();
        else if (vk_submit_and_wait()) goto fail; }
    return hd;
fail:
    vk_destroy_image(&o->im);
    memset(o, 0, sizeof *o);
    return 0;
}

static void vk_eng_texture_upload(void* user, u32 texture, u32 face, u32 mip, u32 w, u32 h,
                                  const void* src, u32 row_bytes, u32 rows)
{
    (void)user;
    vk_eng_obj* o = vk_eng_get(texture, VK_ENG_TEXTURE);
    if (!o || !src || !row_bytes || !rows || face >= (o->cube ? 6u : 1u) || mip >= o->mips) return;
    /* A BC row is a row of 4x4 blocks; the engine's `rows` already counts them. */
    const size_t tight = o->block ? (size_t)((w + 3u) / 4u) * o->block : (size_t)w * o->bpp;
    vk_eng_upload_rows(o->im.img, face, mip, w, h, tight, rows, src, row_bytes);
}

/* 3D textures (a colour-grading LUT: GoW:A's last pass reads a 32x32x32 one;
 * sampled as its first slice it lost all blue and the image went yellow-green).
 * A VK_IMAGE_TYPE_3D image of `depth` slices, `cube` = 2 marking it for the
 * descriptor code, uploaded one slice of decoded rows at a time. */
static u32 vk_eng_texture_create_3d(void* user, rsx_be_format fmt, u32 w, u32 h,
                                    u32 depth, u32 mips, u32 remap, u32 rsx_fmt)
{
    (void)user;
    u32 bpp;
    const VkFormat vf = vk_eng_format(fmt, &bpp);
    const int bc = (fmt == RSX_BE_FMT_BC1 || fmt == RSX_BE_FMT_BC2 || fmt == RSX_BE_FMT_BC3);
    if (vf == VK_FORMAT_UNDEFINED || bc || !w || !h || !depth || w > 2048u || h > 2048u || depth > 2048u)
        return 0;
    if (!mips) mips = 1;
    const u32 hd = vk_eng_alloc();
    if (!hd) return 0;
    vk_eng_obj* o = &s_eobj[hd];
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_3D,
        .format = vf, .extent = { w, h, depth }, .mipLevels = mips, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
    VkMemoryRequirements req;
    u32 type;
    if (pvkCreateImage(s_vk.device, &ici, NULL, &o->im.img) != VK_SUCCESS) goto fail;
    pvkGetImageMemoryRequirements(s_vk.device, o->im.img, &req);
    if (vk_find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &type) &&
        vk_find_memory_type(req.memoryTypeBits, 0, &type)) goto fail;
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = req.size, .memoryTypeIndex = type };
    if (pvkAllocateMemory(s_vk.device, &mai, NULL, &o->im.mem) != VK_SUCCESS ||
        pvkBindImageMemory(s_vk.device, o->im.img, o->im.mem, 0) != VK_SUCCESS ||
        vk_eng_make_view(o->im.img, vf, mips, vk_remap(remap, rsx_fmt), 2, &o->im.view))
        goto fail;
    o->kind = VK_ENG_TEXTURE; o->fmt = vf; o->w = w; o->h = h; o->cube = 2; o->mips = mips;
    o->bpp = bpp; o->block = 0; o->depth3d = depth;
    /* Batched like vk_eng_texture_create's: GoW:A recreates its 32^3 grading
     * LUT whenever it changes (a volume is create-and-swap), ~40 times a second. */
    {   const char* e = getenv("VK_TEX_CREATE_SYNC");
        const int batched = vk_eb_active() && !(e && *e == '1');
        if (batched ? (!s_eb_pending && vk_begin()) : vk_begin()) goto fail;
        vk_barrier_image(o->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
        if (batched) vk_eb_recorded();
        else if (vk_submit_and_wait()) goto fail; }
    return hd;
fail:
    vk_destroy_image(&o->im);
    memset(o, 0, sizeof *o);
    return 0;
}

static void vk_eng_texture_upload_3d(void* user, u32 texture, u32 z, u32 mip,
                                     u32 w, u32 h, const void* src, u32 row_bytes)
{
    (void)user;
    vk_eng_obj* o = vk_eng_get(texture, VK_ENG_TEXTURE);
    if (!o || o->cube != 2 || !src || !row_bytes || mip >= o->mips) return;
    const u32 d = o->depth3d >> mip ? o->depth3d >> mip : 1u;
    if (z >= d) return;
    vk_eng_upload_region_z(o->im.img, 0, mip, 0, 0, z, w, h, (size_t)w * o->bpp, h, src, row_bytes);
}

/* A colour target sampled with a unit's crossbar: an extra view of the same
 * image, cached by (surface, crossbar, format) and released with the target. */
static u32 vk_eng_surface_view(void* user, u32 surface, u32 remap, u32 rsx_format)
{
    (void)user;
    vk_eng_obj* t = vk_eng_get(surface, VK_ENG_COLOR);
    if (!t) return 0;
    for (u32 i = 1; i < VK_ENG_MAX_OBJ; i++)
        if (s_eobj[i].kind == VK_ENG_VIEW && s_eobj[i].surface == surface &&
            s_eobj[i].remap == remap && s_eobj[i].rsx_fmt == rsx_format)
            return i;
    const u32 hd = vk_eng_alloc();
    if (!hd) return 0;
    vk_eng_obj* o = &s_eobj[hd];
    if (vk_eng_make_view(t->im.img, t->fmt, 1, vk_remap(remap, rsx_format), 0, &o->im.view)) {
        memset(o, 0, sizeof *o); return 0;
    }
    o->kind = VK_ENG_VIEW; o->fmt = t->fmt; o->w = t->w; o->h = t->h;
    o->surface = surface; o->remap = remap; o->rsx_fmt = rsx_format;
    return hd;
}
/* ---- colour feedback snapshots ---------------------------------------------
 * A draw that samples a colour target it is also writing (in-place deferred
 * lighting: read the albedo, write the lit colour over it) gets a copy of
 * that target taken just before it. The engine asks for one per such draw
 * (rsx_draw_backend.color_snapshot) and releases the previous one; without
 * this hook it falls back to the guest-memory bytes at the target's address,
 * which hold whatever was last stored there -- for Twisted Metal the previous
 * frame's UI, lit and composited into the next frame.
 *
 * Every operation here is one synchronous submission, so the copy is already
 * in command order: after the draws before it, before the draw that samples
 * it. The copy is a COLOR object marked `snapshot` so the engine can take a
 * crossbar view of it (vk_eng_surface_view); it is never an attachment.
 * Released copies keep their images in a small pool, since a lit frame asks
 * for one per light. */
#define VK_SNAP_POOL 8
static struct { vk_image im; VkFormat fmt; u32 w, h; } s_snap_pool[VK_SNAP_POOL];
static u32 s_snap_pool_n;

static void vk_eng_snapshot_recycle(vk_eng_obj* o)
{
    if (s_snap_pool_n < VK_SNAP_POOL && o->im.img) {
        s_snap_pool[s_snap_pool_n].im = o->im;
        s_snap_pool[s_snap_pool_n].fmt = o->fmt;
        s_snap_pool[s_snap_pool_n].w = o->w;
        s_snap_pool[s_snap_pool_n].h = o->h;
        s_snap_pool_n++;
    } else {
        vk_destroy_image(&o->im);
    }
}

static vk_buffer s_snap_st;         /* depth snapshot bounce buffer (vk_eng_depth_snapshot) */

static void vk_eng_snapshot_pool_drain(void)
{
    for (u32 i = 0; i < s_snap_pool_n; i++) vk_destroy_image(&s_snap_pool[i].im);
    s_snap_pool_n = 0;
    vk_destroy_buffer(&s_snap_st);
}

/* VK_SNAP_FLUSH=<mask> (diagnostic): submit the open batch around the snapshot copies.
 * 1 before a colour copy, 2 after it, 4 before a depth copy, 8 after it. */
static unsigned vk_snap_flush_mask(void)
{
    static int m = -1;
    if (m < 0) { const char* e = getenv("VK_SNAP_FLUSH"); m = e ? atoi(e) : 0; }
    return (unsigned)m;
}
static void vk_snap_flush(unsigned bit) { if ((vk_snap_flush_mask() & bit) && s_eb_pending) vk_eng_flush(); }

static u32 vk_eng_color_snapshot(void* user, u32 surface)
{
    (void)user;
    vk_snap_flush(1);
    const vk_eng_obj* t = vk_eng_get(surface, VK_ENG_COLOR);
    if (!t || t->snapshot) return 0;
    const u32 hd = vk_eng_alloc();
    if (!hd) return 0;
    vk_eng_obj* o = &s_eobj[hd];
    int reused = 0;
    for (u32 i = 0; i < s_snap_pool_n; i++)
        if (s_snap_pool[i].fmt == t->fmt && s_snap_pool[i].w == t->w && s_snap_pool[i].h == t->h) {
            o->im = s_snap_pool[i].im;
            s_snap_pool[i] = s_snap_pool[--s_snap_pool_n];
            reused = 1;
            break;
        }
    if (!reused && vk_create_image(&o->im, t->fmt, t->w, t->h,
                                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                   VK_IMAGE_ASPECT_COLOR_BIT)) {
        vk_destroy_image(&o->im);
        memset(o, 0, sizeof *o);
        return 0;
    }
    o->kind = VK_ENG_COLOR; o->snapshot = 1;
    o->fmt = t->fmt; o->bpp = t->bpp; o->w = t->w; o->h = t->h;
    /* VK_SNAP_BATCH=1: record the copy into the open draw batch, ordered
     * after every draw already in it, instead of a submit and a wait for the
     * GPU per copy (~2.7 ms each in GoW:A's lit passes). */
    static int s_snap_batch = -1;
    if (s_snap_batch < 0) { const char* e = getenv("VK_SNAP_BATCH"); s_snap_batch = e ? atoi(e) : 0; }
    const int batch = s_snap_batch && vk_eb_active();
    if (batch) {
        if (!s_eb_pending && vk_begin()) { vk_eng_release(user, hd); return 0; }
        vk_cmd_after_everything(s_vk.cmd);
    } else if (vk_begin()) { vk_eng_release(user, hd); return 0; }
    vk_barrier_image(o->im.img, VK_IMAGE_ASPECT_COLOR_BIT,
                     reused ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED);
    vk_barrier_image(t->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    VkImageCopy c = {
        .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .extent = { t->w, t->h, 1 },
    };
    pvkCmdCopyImage(s_vk.cmd, t->im.img, VK_IMAGE_LAYOUT_GENERAL,
                    o->im.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    /* The copy, made visible to whatever samples it next. */
    vk_barrier_image(o->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    if (batch) {
        if (++s_eb_pending >= vk_eb_max()) vk_eng_flush();
        vk_snap_flush(2);
        return hd;
    }
    if (vk_submit_and_wait()) { vk_eng_release(user, hd); return 0; }
    return hd;
}

/* ---- E4: depth snapshots --------------------------------------------------
 * A depth target read as a texture, which the engine asks for only after a
 * depth-writing draw. Metal resolves it with a fullscreen pass into R32Float;
 * Vulkan can do it with two copies on the GPU, since a D32_SFLOAT depth
 * aspect copies out as 32-bit floats: depth -> buffer -> an R32_SFLOAT
 * texture. Other depth formats would need a conversion pass: they report 0.
 *
 * Each depth target keeps ONE snapshot texture (its handle in `surface`),
 * refilled on every request, so nothing accumulates however often the engine
 * invalidates and asks again; it goes when the depth target goes. */
static u32 vk_eng_depth_snapshot(void* user, u32 depth, u32 w, u32 h)
{
    vk_snap_flush(4);
    vk_eng_obj* z = vk_eng_get(depth, VK_ENG_DEPTH);
    if (!z) return 0;
    if (s_vk.depth_format != VK_FORMAT_D32_SFLOAT &&
        s_vk.depth_format != VK_FORMAT_D32_SFLOAT_S8_UINT) {
        static int warned_fmt;
        if (!warned_fmt) { VK_LOG("engine: depth snapshots need D32_SFLOAT, this device uses %d\n",
                                  (int)s_vk.depth_format); warned_fmt = 1; }
        return 0;
    }
    if (w > z->w) w = z->w;
    if (h > z->h) h = z->h;
    if (!w || !h) return 0;

    u32 snap = z->surface;
    vk_eng_obj* t = vk_eng_get(snap, VK_ENG_TEXTURE);
    if (!t || t->w != w || t->h != h) {
        if (t) vk_eng_release(user, snap);
        z->surface = 0;
        snap = vk_eng_alloc();
        if (!snap) return 0;
        t = &s_eobj[snap];
        if (vk_create_image(&t->im, VK_FORMAT_R32_SFLOAT, w, h,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            VK_IMAGE_ASPECT_COLOR_BIT)) {
            vk_destroy_image(&t->im); memset(t, 0, sizeof *t); return 0;
        }
        t->kind = VK_ENG_TEXTURE; t->fmt = VK_FORMAT_R32_SFLOAT; t->bpp = 4; t->w = w; t->h = h;
        z = vk_eng_get(depth, VK_ENG_DEPTH);          /* same slot; reload for clarity */
        z->surface = snap;
        if (vk_begin()) return 0;
        vk_barrier_image(t->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
        if (vk_submit_and_wait()) return 0;
    }

    /* The bounce buffer stays (device-local when the device has it, grown as
     * needed): a fresh host buffer per snapshot cost ~4.8 ms each, 375 times
     * in 20 s of God of War: Ascension's first level. With draw batching the
     * copies join the open batch instead of a submission of their own; the
     * draws that sample the snapshot come after them in the same buffer, and
     * each batched draw starts with a full memory barrier. */
    const VkDeviceSize need = (VkDeviceSize)w * h * 4u;
    if (s_snap_st.size < need) {
        vk_destroy_buffer(&s_snap_st);                   /* flushes a pending batch first */
        if (vk_create_device_buffer(&s_snap_st, need,
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
            vk_destroy_buffer(&s_snap_st); return 0;
        }
    }
    vk_buffer st = s_snap_st;
    const int batch = vk_eb_active();
    int rc = (batch && s_eb_pending) ? 0 : vk_begin();
    if (!rc && batch) vk_cmd_after_everything(s_vk.cmd);
    if (!rc) {
        vk_barrier_image(z->im.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
        vk_barrier_image(t->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
        VkBufferImageCopy out = { .imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 },
                                  .imageExtent = { w, h, 1 } };
        pvkCmdCopyImageToBuffer(s_vk.cmd, z->im.img, VK_IMAGE_LAYOUT_GENERAL, st.buf, 1, &out);
        VkBufferMemoryBarrier bb = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = st.buf, .offset = 0, .size = VK_WHOLE_SIZE };
        pvkCmdPipelineBarrier(s_vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, NULL, 1, &bb, 0, NULL);
        VkBufferImageCopy in = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                 .imageExtent = { w, h, 1 } };
        pvkCmdCopyBufferToImage(s_vk.cmd, st.buf, t->im.img, VK_IMAGE_LAYOUT_GENERAL, 1, &in);
        if (batch) {
            vk_barrier_image(t->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
            if (++s_eb_pending >= vk_eb_max()) vk_eng_flush();
        } else
            rc = vk_submit_and_wait();
    }
    if (!rc) vk_snap_flush(8);
    return rc ? 0 : snap;
}

/* A D24S8 depth target read as an A8R8G8B8 texture, filled on the GPU: the
 * depth and stencil aspects are copied into one buffer, a compute pass packs
 * each texel as the guest word would decode (vulkan/rsx_vk_depth_rgba8.comp),
 * and the result is copied into `texture`, an RGBA8 texture of at most the
 * target's size. All of it joins the open batch, so unlike a store to guest
 * memory there is no wait for the GPU and no re-upload. */
static struct {
    int tried, ok;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    vk_buffer in, out;     /* in: cap floats, then cap stencil bytes; out: cap words */
    u32 cap;               /* texels the buffers and the descriptor set are laid out for */
} s_za;

static int vk_za_init(void)
{
    if (s_za.tried) return s_za.ok;
    s_za.tried = 1;
    VkDescriptorSetLayoutBinding b[3];
    for (u32 i = 0; i < 3; i++)
        b[i] = (VkDescriptorSetLayoutBinding){ .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                               .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .bindingCount = 3, .pBindings = b };
    VkPushConstantRange pr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 8 };
    VkPipelineLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                       .setLayoutCount = 1, .pSetLayouts = &s_za.dsl,
                                       .pushConstantRangeCount = 1, .pPushConstantRanges = &pr };
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
    VkShaderModule sm = VK_NULL_HANDLE;
    if (pvkCreateDescriptorSetLayout(s_vk.device, &dci, NULL, &s_za.dsl) != VK_SUCCESS ||
        pvkCreatePipelineLayout(s_vk.device, &lci, NULL, &s_za.layout) != VK_SUCCESS ||
        pvkCreateDescriptorPool(s_vk.device, &pci, NULL, &s_za.pool) != VK_SUCCESS ||
        vk_create_shader(rsx_vk_depth_rgba8_comp_spv, sizeof rsx_vk_depth_rgba8_comp_spv, &sm)) {
        VK_LOG("engine: depth-as-ARGB8 compute setup failed\n");
        return 0;
    }
    VkComputePipelineCreateInfo cci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" },
        .layout = s_za.layout };
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = s_za.pool, .descriptorSetCount = 1,
                                        .pSetLayouts = &s_za.dsl };
    const int ok = pvkCreateComputePipelines(s_vk.device, VK_NULL_HANDLE, 1, &cci, NULL, &s_za.pipe) == VK_SUCCESS &&
                   pvkAllocateDescriptorSets(s_vk.device, &dai, &s_za.set) == VK_SUCCESS;
    pvkDestroyShaderModule(s_vk.device, sm, NULL);
    if (!ok) VK_LOG("engine: depth-as-ARGB8 compute pipeline failed\n");
    s_za.ok = ok;
    return ok;
}

static int vk_eng_depth_to_argb8(void* user, u32 depth, u32 texture, u32* check)
{
    (void)user;
    vk_eng_obj* z = vk_eng_get(depth, VK_ENG_DEPTH);
    vk_eng_obj* t = vk_eng_get(texture, VK_ENG_TEXTURE);
    if (!z || !t || t->fmt != VK_FORMAT_R8G8B8A8_UNORM || t->cube || t->w > z->w || t->h > z->h ||
        (s_vk.depth_format != VK_FORMAT_D32_SFLOAT && s_vk.depth_format != VK_FORMAT_D32_SFLOAT_S8_UINT) ||
        !vk_za_init())
        return -1;
    const u32 w = t->w, h = t->h, n = w * h;
    const int has_stencil = (s_vk.depth_aspect & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
    if (n > s_za.cap) {
        /* Grown at most a few times a run; vk_destroy_buffer idles the GPU first,
         * so nothing in flight still reads the set being rewritten. */
        vk_destroy_buffer(&s_za.in);
        vk_destroy_buffer(&s_za.out);
        s_za.cap = 0;
        const VkDeviceSize cap = ((VkDeviceSize)n + 3u) & ~(VkDeviceSize)3u;
        const VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vk_create_device_buffer(&s_za.in, cap * 5u, u) || vk_create_device_buffer(&s_za.out, cap * 4u, u)) {
            vk_destroy_buffer(&s_za.in); vk_destroy_buffer(&s_za.out);
            return -1;
        }
        VkDescriptorBufferInfo bi[3] = { { s_za.in.buf, 0, cap * 4u }, { s_za.in.buf, cap * 4u, cap },
                                         { s_za.out.buf, 0, cap * 4u } };
        VkWriteDescriptorSet wr[3];
        for (u32 i = 0; i < 3; i++)
            wr[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s_za.set,
                                            .dstBinding = i, .descriptorCount = 1,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            .pBufferInfo = &bi[i] };
        pvkUpdateDescriptorSets(s_vk.device, 3, wr, 0, NULL);
        s_za.cap = (u32)cap;
    }
    const VkDeviceSize soff = (VkDeviceSize)s_za.cap * 4u;   /* where binding 1 starts */

    const int batch = vk_eb_active();
    int rc = (batch && s_eb_pending) ? 0 : vk_begin();
    if (rc) return -1;
    vk_cmd_after_everything(s_vk.cmd);
    vk_barrier_image(z->im.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
    VkBufferImageCopy dc = { .imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 },
                             .imageExtent = { w, h, 1 } };
    pvkCmdCopyImageToBuffer(s_vk.cmd, z->im.img, VK_IMAGE_LAYOUT_GENERAL, s_za.in.buf, 1, &dc);
    if (has_stencil) {
        VkBufferImageCopy sc = { .bufferOffset = soff, .imageSubresource = { VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1 },
                                 .imageExtent = { w, h, 1 } };
        pvkCmdCopyImageToBuffer(s_vk.cmd, z->im.img, VK_IMAGE_LAYOUT_GENERAL, s_za.in.buf, 1, &sc);
    }
    VkMemoryBarrier m1 = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT };
    pvkCmdPipelineBarrier(s_vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 1, &m1, 0, NULL, 0, NULL);
    pvkCmdBindPipeline(s_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_za.pipe);
    pvkCmdBindDescriptorSets(s_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_za.layout, 0, 1, &s_za.set, 0, NULL);
    const u32 pc[2] = { n, (u32)has_stencil };
    pvkCmdPushConstants(s_vk.cmd, s_za.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    pvkCmdDispatch(s_vk.cmd, (n + 63u) / 64u, 1, 1);
    VkMemoryBarrier m2 = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT };
    pvkCmdPipelineBarrier(s_vk.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          0, 1, &m2, 0, NULL, 0, NULL);
    vk_barrier_image(t->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    VkBufferImageCopy oc = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                             .imageExtent = { w, h, 1 } };
    pvkCmdCopyBufferToImage(s_vk.cmd, s_za.out.buf, t->im.img, VK_IMAGE_LAYOUT_GENERAL, 1, &oc);
    vk_buffer* st = NULL;
    if (check && (st = vk_stage_get(&s_rb_stage, (VkDeviceSize)n * 4u, VK_BUFFER_USAGE_TRANSFER_DST_BIT))) {
        VkBufferCopy bc = { 0, 0, (VkDeviceSize)n * 4u };
        pvkCmdCopyBuffer(s_vk.cmd, s_za.out.buf, st->buf, 1, &bc);
        VkMemoryBarrier m3 = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                               .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
        pvkCmdPipelineBarrier(s_vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                              0, 1, &m3, 0, NULL, 0, NULL);
    }
    if (batch && !st) {
        vk_barrier_image(t->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
        if (++s_eb_pending >= vk_eb_max()) vk_eng_flush();
    } else {
        if (batch) ++s_eb_pending;
        rc = batch ? (vk_eng_idle(), 0) : vk_submit_and_wait();
        if (!rc && st) { vk_invalidate(st); memcpy(check, st->ptr, (size_t)n * 4u); }
    }
    return rc ? -1 : 0;
}

/* An RGBA8 colour target read as a Y16_X16_FLOAT texture, filled on the GPU:
 * the target's top-left w x h texels (the texture's size) are copied into a
 * buffer, a compute pass regroups each texel's bytes into the two halves the
 * guest word would decode to (vulkan/rsx_vk_rgba8_rg16.comp), and the result
 * is copied into `texture`, an R16G16F texture. Like vk_eng_depth_to_argb8 it
 * joins the open batch: no wait for the GPU and no re-upload. */
static struct {
    int tried, ok;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    vk_buffer in, out;     /* cap words each */
    u32 cap;
} s_cr;

static int vk_cr_init(void)
{
    if (s_cr.tried) return s_cr.ok;
    s_cr.tried = 1;
    VkDescriptorSetLayoutBinding b[2];
    for (u32 i = 0; i < 2; i++)
        b[i] = (VkDescriptorSetLayoutBinding){ .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                               .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .bindingCount = 2, .pBindings = b };
    VkPushConstantRange pr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 };
    VkPipelineLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                       .setLayoutCount = 1, .pSetLayouts = &s_cr.dsl,
                                       .pushConstantRangeCount = 1, .pPushConstantRanges = &pr };
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
    VkShaderModule sm = VK_NULL_HANDLE;
    if (pvkCreateDescriptorSetLayout(s_vk.device, &dci, NULL, &s_cr.dsl) != VK_SUCCESS ||
        pvkCreatePipelineLayout(s_vk.device, &lci, NULL, &s_cr.layout) != VK_SUCCESS ||
        pvkCreateDescriptorPool(s_vk.device, &pci, NULL, &s_cr.pool) != VK_SUCCESS ||
        vk_create_shader(rsx_vk_rgba8_rg16_comp_spv, sizeof rsx_vk_rgba8_rg16_comp_spv, &sm)) {
        VK_LOG("engine: colour-as-Y16X16F compute setup failed\n");
        return 0;
    }
    VkComputePipelineCreateInfo cci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" },
        .layout = s_cr.layout };
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = s_cr.pool, .descriptorSetCount = 1,
                                        .pSetLayouts = &s_cr.dsl };
    const int ok = pvkCreateComputePipelines(s_vk.device, VK_NULL_HANDLE, 1, &cci, NULL, &s_cr.pipe) == VK_SUCCESS &&
                   pvkAllocateDescriptorSets(s_vk.device, &dai, &s_cr.set) == VK_SUCCESS;
    pvkDestroyShaderModule(s_vk.device, sm, NULL);
    if (!ok) VK_LOG("engine: colour-as-Y16X16F compute pipeline failed\n");
    s_cr.ok = ok;
    return ok;
}

static int vk_eng_color_to_rg16f(void* user, u32 surface, u32 texture, u32* check)
{
    (void)user;
    vk_eng_obj* c = vk_eng_get(surface, VK_ENG_COLOR);
    vk_eng_obj* t = vk_eng_get(texture, VK_ENG_TEXTURE);
    if (!c || !t || c->fmt != VK_FORMAT_R8G8B8A8_UNORM || t->fmt != VK_FORMAT_R16G16_SFLOAT || t->cube ||
        t->w > c->w || t->h > c->h || !vk_cr_init())
        return -1;
    const u32 w = t->w, h = t->h, n = w * h;
    if (n > s_cr.cap) {
        vk_destroy_buffer(&s_cr.in);
        vk_destroy_buffer(&s_cr.out);
        s_cr.cap = 0;
        const VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vk_create_device_buffer(&s_cr.in, (VkDeviceSize)n * 4u, u) ||
            vk_create_device_buffer(&s_cr.out, (VkDeviceSize)n * 4u, u)) {
            vk_destroy_buffer(&s_cr.in); vk_destroy_buffer(&s_cr.out);
            return -1;
        }
        VkDescriptorBufferInfo bi[2] = { { s_cr.in.buf, 0, (VkDeviceSize)n * 4u },
                                         { s_cr.out.buf, 0, (VkDeviceSize)n * 4u } };
        VkWriteDescriptorSet wr[2];
        for (u32 i = 0; i < 2; i++)
            wr[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s_cr.set,
                                            .dstBinding = i, .descriptorCount = 1,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            .pBufferInfo = &bi[i] };
        pvkUpdateDescriptorSets(s_vk.device, 2, wr, 0, NULL);
        s_cr.cap = n;
    }

    const int batch = vk_eb_active();
    int rc = (batch && s_eb_pending) ? 0 : vk_begin();
    if (rc) return -1;
    vk_cmd_after_everything(s_vk.cmd);
    vk_barrier_image(c->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    VkBufferImageCopy ic = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                             .imageExtent = { w, h, 1 } };
    pvkCmdCopyImageToBuffer(s_vk.cmd, c->im.img, VK_IMAGE_LAYOUT_GENERAL, s_cr.in.buf, 1, &ic);
    VkMemoryBarrier m1 = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT };
    pvkCmdPipelineBarrier(s_vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 1, &m1, 0, NULL, 0, NULL);
    pvkCmdBindPipeline(s_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_cr.pipe);
    pvkCmdBindDescriptorSets(s_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_cr.layout, 0, 1, &s_cr.set, 0, NULL);
    pvkCmdPushConstants(s_vk.cmd, s_cr.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n);
    pvkCmdDispatch(s_vk.cmd, (n + 63u) / 64u, 1, 1);
    VkMemoryBarrier m2 = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT };
    pvkCmdPipelineBarrier(s_vk.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          0, 1, &m2, 0, NULL, 0, NULL);
    vk_barrier_image(t->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    VkBufferImageCopy oc = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                             .imageExtent = { w, h, 1 } };
    pvkCmdCopyBufferToImage(s_vk.cmd, s_cr.out.buf, t->im.img, VK_IMAGE_LAYOUT_GENERAL, 1, &oc);
    vk_buffer* st = NULL;
    if (check && (st = vk_stage_get(&s_rb_stage, (VkDeviceSize)n * 4u, VK_BUFFER_USAGE_TRANSFER_DST_BIT))) {
        VkBufferCopy bc = { 0, 0, (VkDeviceSize)n * 4u };
        pvkCmdCopyBuffer(s_vk.cmd, s_cr.out.buf, st->buf, 1, &bc);
        VkMemoryBarrier m3 = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                               .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
        pvkCmdPipelineBarrier(s_vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                              0, 1, &m3, 0, NULL, 0, NULL);
    }
    if (batch && !st) {
        vk_barrier_image(t->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
        if (++s_eb_pending >= vk_eb_max()) vk_eng_flush();
    } else {
        if (batch) ++s_eb_pending;
        rc = batch ? (vk_eng_idle(), 0) : vk_submit_and_wait();
        if (!rc && st) { vk_invalidate(st); memcpy(check, st->ptr, (size_t)n * 4u); }
    }
    return rc ? -1 : 0;
}

/* A w x h rectangle copied between two 32-bit colour images (a colour target
 * or a single-level 2D texture, either side), joining the open batch after
 * every command already in it: no wait for the GPU. `check`, normally NULL,
 * also receives the destination's whole first level (R in the low byte, dst
 * w * h words) after waiting for the GPU. */
static int vk_eng_image_copy(void* user, u32 src, u32 sx, u32 sy, u32 dst, u32 dx, u32 dy,
                             u32 w, u32 h, u32* check)
{
    (void)user;
    vk_eng_obj* s = (src < VK_ENG_MAX_OBJ && (s_eobj[src].kind == VK_ENG_COLOR ||
                                              s_eobj[src].kind == VK_ENG_TEXTURE)) ? &s_eobj[src] : NULL;
    vk_eng_obj* d = vk_eng_get(dst, VK_ENG_TEXTURE);
    if (!s || !d || !w || !h || s->fmt != d->fmt || s->bpp != 4u || d->cube || s->cube ||
        (u64)sx + w > s->w || (u64)sy + h > s->h || (u64)dx + w > d->w || (u64)dy + h > d->h)
        return -1;
    const int batch = vk_eb_active();
    int rc = (batch && s_eb_pending) ? 0 : vk_begin();
    if (rc) return -1;
    vk_cmd_after_everything(s_vk.cmd);
    vk_barrier_image(s->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    vk_barrier_image(d->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    VkImageCopy c = {
        .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .srcOffset = { (int32_t)sx, (int32_t)sy, 0 },
        .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .dstOffset = { (int32_t)dx, (int32_t)dy, 0 },
        .extent = { w, h, 1 },
    };
    pvkCmdCopyImage(s_vk.cmd, s->im.img, VK_IMAGE_LAYOUT_GENERAL, d->im.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    vk_barrier_image(d->im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    vk_buffer* st = NULL;
    const VkDeviceSize bytes = (VkDeviceSize)d->w * d->h * 4u;
    if (check && (st = vk_stage_get(&s_rb_stage, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT))) {
        VkBufferImageCopy bc = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                 .imageExtent = { d->w, d->h, 1 } };
        pvkCmdCopyImageToBuffer(s_vk.cmd, d->im.img, VK_IMAGE_LAYOUT_GENERAL, st->buf, 1, &bc);
        VkMemoryBarrier m3 = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                               .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
        pvkCmdPipelineBarrier(s_vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                              0, 1, &m3, 0, NULL, 0, NULL);
    }
    if (batch && !st) {
        if (++s_eb_pending >= vk_eb_max()) vk_eng_flush();
    } else {
        if (batch) ++s_eb_pending;
        rc = batch ? (vk_eng_idle(), 0) : vk_submit_and_wait();
        if (!rc && st) { vk_invalidate(st); memcpy(check, st->ptr, (size_t)bytes); }
    }
    return rc ? -1 : 0;
}

/* The depth aspect of a D32_SFLOAT target copies out as 32-bit floats, which
 * is what the engine asks for; other depth formats report failure. */
static int vk_eng_depth_readback(void* user, u32 depth, u32 x, u32 y, u32 w, u32 h,
                                 float* out, u32 out_pitch)
{
    (void)user;
    vk_eng_obj* z = vk_eng_get(depth, VK_ENG_DEPTH);
    if (!z || !out || !w || !h || x + w > z->w || y + h > z->h ||
        out_pitch < w * 4u || (s_vk.depth_format != VK_FORMAT_D32_SFLOAT &&
                                  s_vk.depth_format != VK_FORMAT_D32_SFLOAT_S8_UINT))
        return -1;
    vk_buffer* const st = vk_stage_get(&s_rb_stage, (VkDeviceSize)w * h * 4u, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!st) return -1;
    int rc = vk_begin();
    if (!rc) {
        vk_barrier_image(z->im.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
        VkBufferImageCopy region = {
            .imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 },
            .imageOffset = { (int32_t)x, (int32_t)y, 0 }, .imageExtent = { w, h, 1 },
        };
        pvkCmdCopyImageToBuffer(s_vk.cmd, z->im.img, VK_IMAGE_LAYOUT_GENERAL, st->buf, 1, &region);
        VkBufferMemoryBarrier hb = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = st->buf, .offset = 0, .size = VK_WHOLE_SIZE,
        };
        pvkCmdPipelineBarrier(s_vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                              0, 0, NULL, 1, &hb, 0, NULL);
        rc = vk_submit_and_wait();
        if (!rc) {
            vk_invalidate(st);
            for (u32 row = 0; row < h; row++)
                memcpy((u8*)out + (size_t)row * out_pitch,
                       (const u8*)st->ptr + (size_t)row * w * 4u, (size_t)w * 4u);
        }
    }
    return rc ? -1 : 0;
}
/* The stencil aspect copies out as tightly packed bytes (VK_FORMAT_S8_UINT
 * texels) from either combined depth/stencil format. */
static int vk_eng_stencil_readback(void* user, u32 depth, u32 x, u32 y, u32 w, u32 h,
                                   u8* out, u32 out_pitch)
{
    (void)user;
    vk_eng_obj* z = vk_eng_get(depth, VK_ENG_DEPTH);
    if (!z || !out || !w || !h || x + w > z->w || y + h > z->h || out_pitch < w ||
        !(s_vk.depth_aspect & VK_IMAGE_ASPECT_STENCIL_BIT))
        return -1;
    vk_buffer* const st = vk_stage_get(&s_rb_stage, (VkDeviceSize)w * h, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!st) return -1;
    int rc = vk_begin();
    if (!rc) {
        vk_barrier_image(z->im.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_GENERAL);
        VkBufferImageCopy region = {
            .imageSubresource = { VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1 },
            .imageOffset = { (int32_t)x, (int32_t)y, 0 }, .imageExtent = { w, h, 1 },
        };
        pvkCmdCopyImageToBuffer(s_vk.cmd, z->im.img, VK_IMAGE_LAYOUT_GENERAL, st->buf, 1, &region);
        VkBufferMemoryBarrier hb = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = st->buf, .offset = 0, .size = VK_WHOLE_SIZE,
        };
        pvkCmdPipelineBarrier(s_vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                              0, 0, NULL, 1, &hb, 0, NULL);
        rc = vk_submit_and_wait();
        if (!rc) {
            vk_invalidate(st);
            for (u32 row = 0; row < h; row++)
                memcpy(out + (size_t)row * out_pitch, (const u8*)st->ptr + (size_t)row * w, w);
        }
    }
    return rc ? -1 : 0;
}
/* ---- E2: pipelines and draws ---------------------------------------------
 * A pipeline handle carries the two translated modules and the render state
 * the engine keyed it on. Topology is not part of that key (Metal sets it on
 * the encoder), but Vulkan 1.0 bakes it into the pipeline, so each handle
 * builds its VkPipeline per topology on first use.
 *
 * Bindings are the translator's contract (rsx_shader_spirv.h), and the set
 * layout, pipeline layout, descriptor set and constant UBO are the ones the
 * vtable guest path already creates. Vertex inputs are the engine's compact
 * layout: slot n is a float4 at n*16, at location n, as Metal reads it.
 * Known gaps, stated rather than guessed: vertex-texture units (their
 * samplers s0..s3 would alias the pixel samplers' binding), colour targets
 * other than RGBA8 and RGBA16F, depth-only passes, and stencil (the depth format
 * has no stencil aspect yet, so the test is inert). */
/* As many as the engine keeps (ENG_MAX_PIPELINES in rsx_draw_engine.c). God of
 * War: Ascension selects ~740 distinct fragment programs in its first level on
 * top of the menu's; with 512 slots every later pipeline failed silently and the
 * level drew nothing but its clears. */
#define VK_ENG_MAX_PIPE   8192
#define VK_ENG_TOPOLOGIES 5
#define VK_ENG_FB_CACHE   64

typedef struct vk_eng_pipe {
    int                 used;
    VkShaderModule      vs, fs;
    rsx_be_render_state rs;
    u32                 nslots, stride, rt_count;
    VkFormat            rt_fmt;          /* picks the render pass it is built against */
    u32                 cube_mask;       /* TextureCube and Texture3D units: the split set layout */
    u32                 vol_mask;        /* of those, the Texture3D ones */
    VkPipeline          variant[VK_ENG_TOPOLOGIES];
    /* Asynchronous builds (vk_eng_async): the two programs' text until their
     * modules exist; state 0 modules pending, 1 ready, -1 failed; topologies
     * asked for and settled (built or failed); a worker owns it while running. */
    char*               vs_text;
    char*               ps_text;
    int                 state;
    u32                 want, done;
    int                 running;
} vk_eng_pipe;

static vk_eng_pipe s_epipe[VK_ENG_MAX_PIPE];

static struct {
    VkRenderPass rp[VK_ENG_RT_FORMATS][RSX_BE_MAX_COLOR_TARGETS + 1]; /* by format, colour count */
    struct { u32 key[RSX_BE_MAX_COLOR_TARGETS + 1]; VkFramebuffer fb; } fb[VK_ENG_FB_CACHE];
    u32          fb_count;
    vk_image     scratch_depth;                          /* when no zeta is bound */
    u32          scratch_w, scratch_h;
    u32          pipeline;
    u32          vs_bytes, ps_bytes;
    int          cb_dirty;      /* constants bound since the last upload into a ring slice */
    u8*          ucpu;                /* the constants, in the shared UBO's layout */
    u32          tex[RSX_BE_MAX_TEXTURES];
    rsx_be_sampler_desc samp[RSX_BE_MAX_TEXTURES];
    u32          tex_mask;
    float        vp[4];
    u32          sc[4];
    int          have_vp, have_sc;
    u32          stencil_ref;
    float        blend_color[4];      /* SET_BLEND_COLOR for the next draw */
    float        depth_bounds[2];     /* min, max for the next draw */
    float        depth_bias[2];       /* slope factor, constant units for the next draw */
    vk_buffer    indices;
    u32          warned;
    /* The split set layout (vk_eng_split_layout) and its one set, made the
     * first time a pipeline samples a cube map; the 1x1 cube a cube unit
     * with nothing bound samples. */
    VkDescriptorSetLayout split_layout;
    VkDescriptorPool      split_pool;
    VkDescriptorSet       split_set;
    VkPipelineLayout      split_pipe_layout;
    u32                   dummy_cube;
    u32                   dummy_vol;
} s_e2;

#define VK_ENG_SAMP_CACHE 64
static struct { u64 key; VkSampler s; } s_esamp[VK_ENG_SAMP_CACHE];
static u32 s_esamp_count;

static void vk_eng_drop_framebuffers(void)
{
    vk_eng_idle();
    for (u32 i = 0; i < s_e2.fb_count; i++)
        if (s_e2.fb[i].fb) pvkDestroyFramebuffer(s_vk.device, s_e2.fb[i].fb, NULL);
    s_e2.fb_count = 0;
}

/* The render pass for `n` colour attachments of format `cf` plus depth,
 * LOAD/STORE in GENERAL like the vtable path's: clears are separate
 * operations. An MRT set shares target A's format (engine contract). */
static VkRenderPass vk_eng_render_pass(VkFormat cf, u32 n)
{
    const int fi = vk_eng_rt_index(cf);
    if (fi < 0 || n < 1 || n > RSX_BE_MAX_COLOR_TARGETS) return VK_NULL_HANDLE;
    if (s_e2.rp[fi][n]) return s_e2.rp[fi][n];
    VkAttachmentDescription att[RSX_BE_MAX_COLOR_TARGETS + 1];
    VkAttachmentReference   cref[RSX_BE_MAX_COLOR_TARGETS];
    for (u32 i = 0; i <= n; i++) {
        att[i] = (VkAttachmentDescription){
            .format = (i < n) ? cf : s_vk.depth_format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            /* Stencil lives across draws like depth does (a title writes it in
             * one pass and tests it in the next); ignored for colour and for
             * depth formats without stencil. */
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
            .initialLayout = VK_IMAGE_LAYOUT_GENERAL, .finalLayout = VK_IMAGE_LAYOUT_GENERAL };
        if (i < n) cref[i] = (VkAttachmentReference){ i, VK_IMAGE_LAYOUT_GENERAL };
    }
    VkAttachmentReference dref = { n, VK_IMAGE_LAYOUT_GENERAL };
    VkSubpassDescription sub = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 .colorAttachmentCount = n, .pColorAttachments = cref,
                                 .pDepthStencilAttachment = &dref };
    VkRenderPassCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                                   .attachmentCount = n + 1, .pAttachments = att,
                                   .subpassCount = 1, .pSubpasses = &sub };
    if (pvkCreateRenderPass(s_vk.device, &rci, NULL, &s_e2.rp[fi][n]) != VK_SUCCESS)
        s_e2.rp[fi][n] = VK_NULL_HANDLE;
    return s_e2.rp[fi][n];
}

/* Tables carried from the Metal backend, itself copied from the D3D12 live
 * draw engine a title has shipped on. */
static VkBlendFactor vk_gcm_blend_factor(u32 f)
{
    switch (f) {
    case 0x0000: return VK_BLEND_FACTOR_ZERO;
    case 0x0001: return VK_BLEND_FACTOR_ONE;
    case 0x0300: return VK_BLEND_FACTOR_SRC_COLOR;
    case 0x0301: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 0x0302: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 0x0303: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 0x0304: return VK_BLEND_FACTOR_DST_ALPHA;
    case 0x0305: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 0x0306: return VK_BLEND_FACTOR_DST_COLOR;
    case 0x0307: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 0x0308: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    /* GL_CONSTANT_* (SET_BLEND_COLOR, a dynamic state here): they fell to ONE, which
     * added GoW:A's bloom layers at full weight. */
    case 0x8001: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 0x8002: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 0x8003: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 0x8004: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default:     return VK_BLEND_FACTOR_ONE;
    }
}
static VkBlendOp vk_gcm_blend_op(u32 e)
{
    switch (e) {
    case 0x8007: return VK_BLEND_OP_MIN;
    case 0x8008: return VK_BLEND_OP_MAX;
    case 0x800A: return VK_BLEND_OP_SUBTRACT;
    case 0x800B: return VK_BLEND_OP_REVERSE_SUBTRACT;
    default:     return VK_BLEND_OP_ADD;                  /* FUNC_ADD 0x8006 */
    }
}
static VkCompareOp vk_gcm_compare(u32 f)
{
    return (f >= 0x0200u && f <= 0x0206u) ? (VkCompareOp)(f - 0x0200u) : VK_COMPARE_OP_ALWAYS;
}
static VkStencilOp vk_gcm_stencil_op(u32 op)
{
    switch (op) {
    case 0x0000: return VK_STENCIL_OP_ZERO;
    case 0x1E01: return VK_STENCIL_OP_REPLACE;
    case 0x1E02: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
    case 0x1E03: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
    case 0x150A: return VK_STENCIL_OP_INVERT;
    case 0x8507: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
    case 0x8508: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
    default:     return VK_STENCIL_OP_KEEP;               /* GL_KEEP 0x1E00 */
    }
}
static VkStencilOpState vk_gcm_stencil_face(u32 func, u32 fail, u32 zfail, u32 zpass,
                                            u32 cmp_mask, u32 write_mask)
{
    VkStencilOpState s = { .failOp = vk_gcm_stencil_op(fail), .passOp = vk_gcm_stencil_op(zpass),
                           .depthFailOp = vk_gcm_stencil_op(zfail),
                           .compareOp = vk_gcm_compare(func),
                           .compareMask = cmp_mask & 0xFFu, .writeMask = write_mask & 0xFFu,
                           .reference = 0 };                /* dynamic */
    return s;
}

static int vk_eng_translate(const char* hlsl, int stage, VkShaderModule* out)
{
    /* Per thread: the compile workers translate too (vk_eng_async). */
    static _Thread_local u32* spv;
    static _Thread_local char log[1024];
    if (!spv && !(spv = (u32*)malloc(VK_SPV_WORDS * sizeof(u32)))) return -1;
    u32 words = 0;
    if (rsx_hlsl_to_spirv(hlsl, stage, spv, VK_SPV_WORDS, &words, log, sizeof log) || !words) {
        VK_LOG("engine: %s translation failed: %s\n",
               stage == RSX_SHADER_STAGE_VERTEX ? "vertex" : "fragment", log);
        return -1;
    }
    VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                    .codeSize = (size_t)words * 4u, .pCode = spv };
    return pvkCreateShaderModule(s_vk.device, &ci, NULL, out) == VK_SUCCESS ? 0 : -1;
}

/* One VkShaderModule per distinct program text (VK_ENG_SHARE_MODULES=0: one
 * per pipeline, as before). The engine makes a pipeline per program pair AND
 * render state (blend, depth, stencil, cull), so the same two programs come
 * back in dozens of pipelines; each used to bring its own modules, and
 * MoltenVK, which caches its SPIR-V -> MSL translation and the compiled Metal
 * library inside the module, translated and compiled the same program again
 * for every one. Shared modules live until shutdown. */
#define VK_ENG_MODTAB 16384u
static struct { u64 h; u32 len; int stage; VkShaderModule m; } s_emod[VK_ENG_MODTAB];
static u32 s_emod_n;
static pthread_mutex_t s_emod_mu = PTHREAD_MUTEX_INITIALIZER;   /* the workers share the table */
static int vk_eng_share_modules(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("VK_ENG_SHARE_MODULES"); on = !(e && *e == '0'); }
    return on;
}
static int vk_eng_module_locked(const char* hlsl, int stage, VkShaderModule* out);
static int vk_eng_module(const char* hlsl, int stage, VkShaderModule* out)
{
    if (!vk_eng_share_modules()) return vk_eng_translate(hlsl, stage, out);
    pthread_mutex_lock(&s_emod_mu);
    const int r = vk_eng_module_locked(hlsl, stage, out);
    pthread_mutex_unlock(&s_emod_mu);
    return r;
}
static int vk_eng_module_locked(const char* hlsl, int stage, VkShaderModule* out)
{
    const u32 len = (u32)strlen(hlsl);
    u64 h = 1469598103934665603ull ^ (u64)stage;
    for (u32 i = 0; i < len; i++) { h ^= (u8)hlsl[i]; h *= 1099511628211ull; }
    u32 ix = (u32)(h ^ (h >> 31)) & (VK_ENG_MODTAB - 1u);
    for (; s_emod[ix].m; ix = (ix + 1u) & (VK_ENG_MODTAB - 1u))
        if (s_emod[ix].h == h && s_emod[ix].len == len && s_emod[ix].stage == stage) {
            *out = s_emod[ix].m;
            return 0;
        }
    if (s_emod_n >= VK_ENG_MODTAB / 2u) return vk_eng_translate(hlsl, stage, out);   /* table full: unshared */
    if (vk_eng_translate(hlsl, stage, out)) return -1;
    s_emod[ix].h = h; s_emod[ix].len = len; s_emod[ix].stage = stage; s_emod[ix].m = *out;
    s_emod_n++;
    return 0;
}
static int vk_eng_module_shared(VkShaderModule m)
{
    if (!m || !vk_eng_share_modules()) return 0;
    int r = 0;
    pthread_mutex_lock(&s_emod_mu);
    for (u32 i = 0; i < VK_ENG_MODTAB && !r; i++) if (s_emod[i].m == m) r = 1;
    pthread_mutex_unlock(&s_emod_mu);
    return r;
}
static void vk_eng_modules_destroy(void)
{
    for (u32 i = 0; i < VK_ENG_MODTAB; i++)
        if (s_emod[i].m) pvkDestroyShaderModule(s_vk.device, s_emod[i].m, NULL);
    memset(s_emod, 0, sizeof s_emod);
    s_emod_n = 0;
}

/* A pixel program with a cube unit declares all sixteen units one by one
 * (TextureCube or Texture2D rsx_texN : register(tN)) instead of the
 * Texture2D array, and register(tN) is binding RSX_SPIRV_TEXTURE_BINDING + N
 * (rsx_shader_spirv.h): sixteen single bindings, where the shared layout has
 * one binding of sixteen. Those pipelines use this layout, made on first use
 * so a title without cube maps never builds it. */
static VkPipelineLayout vk_eng_split_layout(void)
{
    if (s_e2.split_pipe_layout) return s_e2.split_pipe_layout;
    const VkShaderStageFlags both = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding b[3 + RSX_MAX_TEXTURES];
    b[0] = (VkDescriptorSetLayoutBinding){ RSX_SPIRV_VPCONST_BINDING, VK_UB_TYPE, 1, both, NULL };
    b[1] = (VkDescriptorSetLayoutBinding){ RSX_SPIRV_PSCONST_BINDING, VK_UB_TYPE, 1, both, NULL };
    b[2] = (VkDescriptorSetLayoutBinding){ RSX_SPIRV_SAMPLER_BINDING, VK_DESCRIPTOR_TYPE_SAMPLER,
                                           RSX_MAX_TEXTURES, VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
    for (u32 u = 0; u < RSX_MAX_TEXTURES; u++)
        b[3 + u] = (VkDescriptorSetLayoutBinding){ RSX_SPIRV_TEXTURE_BINDING + u,
                                                   VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                                                   VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
    VkDescriptorSetLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .bindingCount = 3 + RSX_MAX_TEXTURES, .pBindings = b };
    VkDescriptorPoolSize ps[3] = {
        { VK_UB_TYPE, 2 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, RSX_MAX_TEXTURES },
        { VK_DESCRIPTOR_TYPE_SAMPLER, RSX_MAX_TEXTURES },
    };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = 1, .poolSizeCount = 3, .pPoolSizes = ps };
    if (pvkCreateDescriptorSetLayout(s_vk.device, &lci, NULL, &s_e2.split_layout) != VK_SUCCESS ||
        pvkCreateDescriptorPool(s_vk.device, &pci, NULL, &s_e2.split_pool) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = s_e2.split_pool, .descriptorSetCount = 1,
                                        .pSetLayouts = &s_e2.split_layout };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1, .pSetLayouts = &s_e2.split_layout };
    if (pvkAllocateDescriptorSets(s_vk.device, &dai, &s_e2.split_set) != VK_SUCCESS ||
        pvkCreatePipelineLayout(s_vk.device, &plci, NULL, &s_e2.split_pipe_layout) != VK_SUCCESS) {
        s_e2.split_pipe_layout = VK_NULL_HANDLE;
        return VK_NULL_HANDLE;
    }
    return s_e2.split_pipe_layout;
}

/* Which units a pixel program declares as Texture3D (rsx_fp_decompiler.c). */
static u32 vk_eng_vol_units(const char* ps_hlsl)
{
    u32 mask = 0;
    char decl[40];
    for (u32 u = 0; u < RSX_MAX_TEXTURES; u++) {
        snprintf(decl, sizeof decl, "Texture3D   rsx_tex%u ", u);
        if (strstr(ps_hlsl, decl)) mask |= 1u << u;
    }
    return mask;
}

/* Which units a pixel program declares as TextureCube. */
static u32 vk_eng_cube_units(const char* ps_hlsl)
{
    u32 mask = 0;
    char decl[40];
    for (u32 u = 0; u < RSX_MAX_TEXTURES; u++) {
        snprintf(decl, sizeof decl, "TextureCube rsx_tex%u ", u);
        if (strstr(ps_hlsl, decl)) mask |= 1u << u;
    }
    return mask;
}

static int  vk_ea_on(void);
static void vk_ea_push(u32 slot);
static u32 vk_eng_pipeline_create(void* user, const char* vs_hlsl, const char* ps_hlsl,
                                  const rsx_be_render_state* rs, const rsx_vertex_layout_plan* layout,
                                  u32 vertex_stride, rsx_be_format rt_fmt, u32 rt_count)
{
    (void)user;
    /* Same gate as Metal: no guest programs, no engine pipelines. */
    if (!s_g.on || !vs_hlsl || !ps_hlsl || !rs || !layout || !vertex_stride) return 0;
    u32 rt_bpp;
    const VkFormat cf = vk_eng_format(rt_fmt, &rt_bpp);
    if (vk_eng_rt_index(cf) < 0) {
        if (!(s_e2.warned & 1u)) { VK_LOG("engine: colour format %d not supported yet\n", (int)rt_fmt); s_e2.warned |= 1u; }
        return 0;
    }
    if (!rt_count) rt_count = 1;
    if (rt_count > RSX_BE_MAX_COLOR_TARGETS) rt_count = RSX_BE_MAX_COLOR_TARGETS;
    u32 slot = VK_ENG_MAX_PIPE;
    for (u32 i = 0; i < VK_ENG_MAX_PIPE; i++) if (!s_epipe[i].used) { slot = i; break; }
    if (slot == VK_ENG_MAX_PIPE) {
        if (!(s_e2.warned & 128u)) {
            VK_LOG("engine: pipeline table full (%u), draws will drop\n", (unsigned)VK_ENG_MAX_PIPE);
            s_e2.warned |= 128u;
        }
        return 0;
    }
    vk_eng_pipe* P = &s_epipe[slot];
    memset(P, 0, sizeof *P);
    if (vk_ea_on()) {
        /* Made here, on the FIFO thread, so a worker only looks them up. */
        P->vol_mask = vk_eng_vol_units(ps_hlsl);
        P->cube_mask = vk_eng_cube_units(ps_hlsl) | P->vol_mask;
        if (!vk_eng_render_pass(cf, rt_count) || (P->cube_mask && !vk_eng_split_layout())) return 0;
        P->vs_text = strdup(vs_hlsl);
        P->ps_text = strdup(ps_hlsl);
        if (!P->vs_text || !P->ps_text) { free(P->vs_text); free(P->ps_text); memset(P, 0, sizeof *P); return 0; }
        P->rs = *rs;
        P->nslots = layout->count;
        P->stride = vertex_stride;
        P->rt_count = rt_count;
        P->rt_fmt = cf;
        __atomic_store_n(&P->used, 1, __ATOMIC_RELEASE);
        vk_ea_push(slot);                  /* the modules, before any draw asks */
        return slot + 1u;
    }
    if (vk_eng_module(vs_hlsl, RSX_SHADER_STAGE_VERTEX, &P->vs)) return 0;
    if (vk_eng_module(ps_hlsl, RSX_SHADER_STAGE_FRAGMENT, &P->fs)) {
        if (!vk_eng_module_shared(P->vs)) pvkDestroyShaderModule(s_vk.device, P->vs, NULL);
        memset(P, 0, sizeof *P); return 0;
    }
    P->rs = *rs;
    P->nslots = layout->count;
    P->stride = vertex_stride;
    P->rt_count = rt_count;
    P->rt_fmt = cf;
    P->vol_mask = vk_eng_vol_units(ps_hlsl);
    P->cube_mask = vk_eng_cube_units(ps_hlsl) | P->vol_mask;
    P->state = 1;
    P->used = 1;
    return slot + 1u;
}

static void vk_eng_pipeline_release(void* user, u32 pipeline)
{
    (void)user;
    vk_eng_idle();
    if (!pipeline || pipeline > VK_ENG_MAX_PIPE || !s_epipe[pipeline - 1].used) return;
    vk_eng_pipe* P = &s_epipe[pipeline - 1];
    /* A worker may hold it: take it over before tearing it down. */
    while (__atomic_exchange_n(&P->running, 1, __ATOMIC_ACQ_REL)) sched_yield();
    __atomic_store_n(&P->used, 0, __ATOMIC_RELEASE);
    free(P->vs_text); free(P->ps_text);
    for (u32 t = 0; t < VK_ENG_TOPOLOGIES; t++)
        if (P->variant[t]) pvkDestroyPipeline(s_vk.device, P->variant[t], NULL);
    if (P->vs && !vk_eng_module_shared(P->vs)) pvkDestroyShaderModule(s_vk.device, P->vs, NULL);
    if (P->fs && !vk_eng_module_shared(P->fs)) pvkDestroyShaderModule(s_vk.device, P->fs, NULL);
    memset(P, 0, sizeof *P);
    if (s_e2.pipeline == pipeline) s_e2.pipeline = 0;
}

/* vkCreateGraphicsPipelines for one topology of P, on whichever thread asks:
 * the render pass and the split layout it reads were made on the FIFO thread
 * when P was (vk_eng_pipeline_create), so here they are only looked up. */
static VkPipeline vk_eng_variant_build(vk_eng_pipe* P, u32 topo_idx)
{
    static const VkPrimitiveTopology topo[VK_ENG_TOPOLOGIES] = {
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
        VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP };
    VkRenderPass rp = vk_eng_render_pass(P->rt_fmt, P->rt_count);
    if (!rp || P->nslots > VK_GUEST_ATTRS) return VK_NULL_HANDLE;
    const VkPipelineLayout layout = P->cube_mask ? vk_eng_split_layout() : s_g.pipe_layout;
    if (!layout) return VK_NULL_HANDLE;
    const rsx_be_render_state* rs = &P->rs;

    VkPipelineShaderStageCreateInfo st[2] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = P->vs, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = P->fs, .pName = "main" } };
    VkVertexInputBindingDescription vb = { 0, P->stride, VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription va[VK_GUEST_ATTRS];
    for (u32 s = 0; s < P->nslots; s++)
        va[s] = (VkVertexInputAttributeDescription){ s, 0, VK_FORMAT_R32G32B32A32_SFLOAT, s * 16u };
    VkPipelineVertexInputStateCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vb,
        .vertexAttributeDescriptionCount = P->nslots, .pVertexAttributeDescriptions = va };
    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = topo[topo_idx] };
    VkPipelineViewportStateCreateInfo vps = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1 };
    /* CULL_FACE FRONT 0x0404 / BACK 0x0405 / FRONT_AND_BACK 0x0408 (front, as
     * in D3D12 and Metal); FRONT_FACE CCW 0x0901 maps directly: both APIs
     * judge winding on screen, and the Y flip in the vertex stage makes this
     * screen the same one D3D's viewport produces. */
    VkPipelineRasterizationStateCreateInfo ras = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = !rs->cull_enable ? VK_CULL_MODE_NONE
                  : (rs->cull_face == 0x0404u || rs->cull_face == 0x0408u) ? VK_CULL_MODE_FRONT_BIT
                                                                           : VK_CULL_MODE_BACK_BIT,
        .frontFace = (rs->front_face == 0x0901u) ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                 : VK_FRONT_FACE_CLOCKWISE,
        /* Always on, its values dynamic (vk_eng_set_depth_bias): 0, 0 is no offset. */
        .depthBiasEnable = VK_TRUE,
        .lineWidth = 1.0f };
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkStencilOpState front = vk_gcm_stencil_face(rs->s_func, rs->s_fail, rs->s_zfail, rs->s_zpass,
                                                 rs->s_func_mask, rs->s_write_mask);
    VkStencilOpState back = rs->stencil_two_sided
        ? vk_gcm_stencil_face(rs->bs_func, rs->bs_fail, rs->bs_zfail, rs->bs_zpass,
                              rs->s_func_mask, rs->s_write_mask)
        : front;
    VkPipelineDepthStencilStateCreateInfo ds = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = rs->depth_test ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = (rs->depth_test && rs->depth_write) ? VK_TRUE : VK_FALSE,
        .depthCompareOp = vk_gcm_compare(rs->depth_func),
        .stencilTestEnable = rs->stencil_enable ? VK_TRUE : VK_FALSE,
        .front = front, .back = back,
        /* The bounds themselves are dynamic (vk_eng_set_depth_bounds). */
        .depthBoundsTestEnable = (rs->depth_bounds_test && s_vk.depth_bounds_ok) ? VK_TRUE : VK_FALSE,
        .minDepthBounds = 0.0f, .maxDepthBounds = 1.0f };
    if (rs->depth_bounds_test && !s_vk.depth_bounds_ok && !(s_e2.warned & 64u)) {
        VK_LOG("engine: the device has no depthBounds; the depth bounds test is ignored\n");
        s_e2.warned |= 64u;
    }
    /* nv40 COLOR_MASK: B [0:7], G [8:15], R [16:23], A [24:31], any bit set.
     * Every attachment takes target A's blend and mask (see rsx_draw_engine.h). */
    VkColorComponentFlags wm = 0;
    if ((rs->color_mask >> 16) & 0xFFu) wm |= VK_COLOR_COMPONENT_R_BIT;
    if ((rs->color_mask >>  8) & 0xFFu) wm |= VK_COLOR_COMPONENT_G_BIT;
    if ((rs->color_mask >>  0) & 0xFFu) wm |= VK_COLOR_COMPONENT_B_BIT;
    if ((rs->color_mask >> 24) & 0xFFu) wm |= VK_COLOR_COMPONENT_A_BIT;
    /* VK_ENG_DEBUG_NODS=1 (diagnostic): no depth/stencil test, no culling, all
     * colour channels written -- tells "the draws rasterize but a test rejects
     * them" from "the draws produce no fragments at all". */
    /* A value is a bit mask: 1 depth, 2 stencil, 4 culling, 8 colour mask
     * (1 alone = "1"; any other non-numeric value = all of them). */
    { static int s_nods = -1;
      if (s_nods < 0) { const char* e = getenv("VK_ENG_DEBUG_NODS");
                        s_nods = !e ? 0 : (atoi(e) > 0 ? atoi(e) : 15); }
      if (s_nods & 1) { ds.depthTestEnable = VK_FALSE; ds.depthWriteEnable = VK_FALSE; }
      if (s_nods & 2) ds.stencilTestEnable = VK_FALSE;
      if (s_nods & 4) ras.cullMode = VK_CULL_MODE_NONE;
      if (s_nods & 8)
          wm = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      /* 16: swap the front-face winding (diagnostic for a Y-flip mismatch). */
      if (s_nods & 16)
          ras.frontFace = (ras.frontFace == VK_FRONT_FACE_CLOCKWISE) ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                                     : VK_FRONT_FACE_CLOCKWISE; }
    VkPipelineColorBlendAttachmentState cba[RSX_BE_MAX_COLOR_TARGETS];
    for (u32 r = 0; r < P->rt_count; r++)
        cba[r] = (VkPipelineColorBlendAttachmentState){
            .blendEnable = (rs->blend_enable && P->rt_fmt != VK_FORMAT_R32_SFLOAT) ? VK_TRUE : VK_FALSE,
            .srcColorBlendFactor = vk_gcm_blend_factor(rs->sf_rgb),
            .dstColorBlendFactor = vk_gcm_blend_factor(rs->df_rgb),
            .colorBlendOp = vk_gcm_blend_op(rs->eq_rgb),
            .srcAlphaBlendFactor = vk_gcm_blend_factor(rs->sf_a),
            .dstAlphaBlendFactor = vk_gcm_blend_factor(rs->df_a),
            .alphaBlendOp = vk_gcm_blend_op(rs->eq_a),
            .colorWriteMask = wm };
    VkPipelineColorBlendStateCreateInfo cb = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = P->rt_count, .pAttachments = cba };
    VkDynamicState dyn[6] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                              VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                              VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_DEPTH_BOUNDS };
    VkPipelineDynamicStateCreateInfo dy = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = s_vk.depth_bounds_ok ? 6u : 5u, .pDynamicStates = dyn };
    VkGraphicsPipelineCreateInfo gci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &vps, .pRasterizationState = &ras, .pMultisampleState = &ms,
        .pDepthStencilState = &ds, .pColorBlendState = &cb, .pDynamicState = &dy,
        .layout = layout, .renderPass = rp, .subpass = 0 };
    VkPipeline out = VK_NULL_HANDLE;
    if (pvkCreateGraphicsPipelines(s_vk.device, VK_NULL_HANDLE, 1, &gci, NULL, &out) != VK_SUCCESS) {
        VK_LOG("engine: vkCreateGraphicsPipelines failed\n");
        out = VK_NULL_HANDLE;
    }
    return out;
}

/* ---- asynchronous pipeline builds ------------------------------------------
 * A program pair the game had never shown cost its first frame the whole host
 * compile: glslang (HLSL -> SPIR-V) in pipeline_create, then MoltenVK's MSL
 * translation and the Metal compiler in vkCreateGraphicsPipelines -- 30 to
 * 190 ms per pipeline, measured in God of War: Ascension, and a new area
 * brings a hundred of them, a freeze of seconds. With VK_ENG_ASYNC=N (default
 * 2; 0 = build on the FIFO thread, as before) pipeline_create only records the
 * text and returns the handle; N worker threads make the modules and the
 * topology variants, and a draw whose variant is not ready yet is skipped
 * (pipeline_ready), so the object appears a few frames late instead of the
 * frame stopping. Vulkan lets pipelines and modules be created on any thread;
 * glslang runs under its own lock (rsx_shader_spirv.cpp). */
#ifndef _WIN32
#define VK_ENG_JOBQ 16384u
static struct {
    int             n;              /* workers; 0 = synchronous */
    pthread_t       th[8];
    pthread_mutex_t mu;
    pthread_cond_t  cv, idle;
    u32             q[VK_ENG_JOBQ];
    u32             head, tail;     /* q[head % N] .. q[tail % N] */
    int             quit, busy;
    u64             built, failed;
    double          ms, max_ms;
} s_ea;

static double vk_ea_ms(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* One pass over P's outstanding work, by the one worker that owns it. */
static void vk_ea_work(vk_eng_pipe* P)
{
    if (!__atomic_load_n(&P->used, __ATOMIC_ACQUIRE)) return;
    if (__atomic_load_n(&P->state, __ATOMIC_ACQUIRE) == 0) {
        const double t0 = vk_ea_ms();
        VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
        int ok = !vk_eng_module(P->vs_text, RSX_SHADER_STAGE_VERTEX, &vs);
        if (ok && vk_eng_module(P->ps_text, RSX_SHADER_STAGE_FRAGMENT, &fs)) {
            if (!vk_eng_module_shared(vs)) pvkDestroyShaderModule(s_vk.device, vs, NULL);
            ok = 0;
        }
        free(P->vs_text); free(P->ps_text);
        P->vs_text = P->ps_text = NULL;
        P->vs = ok ? vs : VK_NULL_HANDLE;
        P->fs = ok ? fs : VK_NULL_HANDLE;
        __atomic_store_n(&P->state, ok ? 1 : -1, __ATOMIC_RELEASE);
        const double ms = vk_ea_ms() - t0;
        pthread_mutex_lock(&s_ea.mu);
        s_ea.ms += ms; if (ms > s_ea.max_ms) s_ea.max_ms = ms;
        if (!ok) s_ea.failed++;
        pthread_mutex_unlock(&s_ea.mu);
    }
    const int ok = __atomic_load_n(&P->state, __ATOMIC_ACQUIRE) == 1;
    for (;;) {
        const u32 todo = __atomic_load_n(&P->want, __ATOMIC_ACQUIRE) & ~__atomic_load_n(&P->done, __ATOMIC_ACQUIRE);
        if (!todo) break;
        const u32 t = (u32)__builtin_ctz(todo);
        const double t0 = vk_ea_ms();
        const VkPipeline v = ok ? vk_eng_variant_build(P, t) : VK_NULL_HANDLE;
        const double ms = vk_ea_ms() - t0;
        if (v) __atomic_store_n(&P->variant[t], v, __ATOMIC_RELEASE);
        __atomic_fetch_or(&P->done, 1u << t, __ATOMIC_ACQ_REL);
        pthread_mutex_lock(&s_ea.mu);
        if (v) s_ea.built++; else if (ok) s_ea.failed++;
        s_ea.ms += ms; if (ms > s_ea.max_ms) s_ea.max_ms = ms;
        pthread_mutex_unlock(&s_ea.mu);
    }
}

static void* vk_ea_main(void* arg)
{
    (void)arg;
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
    pthread_mutex_lock(&s_ea.mu);
    for (;;) {
        while (!s_ea.quit && s_ea.head == s_ea.tail) {
            if (!s_ea.busy) pthread_cond_broadcast(&s_ea.idle);
            pthread_cond_wait(&s_ea.cv, &s_ea.mu);
        }
        if (s_ea.quit) break;
        const u32 slot = s_ea.q[s_ea.head++ % VK_ENG_JOBQ];
        s_ea.busy++;
        pthread_mutex_unlock(&s_ea.mu);
        vk_eng_pipe* P = &s_epipe[slot];
        /* Own P, work, let go, and look again: a request that came while we
         * held it found it owned and left the work to us. */
        while (!__atomic_exchange_n(&P->running, 1, __ATOMIC_ACQ_REL)) {
            vk_ea_work(P);
            __atomic_store_n(&P->running, 0, __ATOMIC_RELEASE);
            if (!__atomic_load_n(&P->used, __ATOMIC_ACQUIRE) ||
                (__atomic_load_n(&P->state, __ATOMIC_ACQUIRE) != 0 &&
                 !(__atomic_load_n(&P->want, __ATOMIC_ACQUIRE) & ~__atomic_load_n(&P->done, __ATOMIC_ACQUIRE))))
                break;
        }
        pthread_mutex_lock(&s_ea.mu);
        s_ea.busy--;
    }
    pthread_mutex_unlock(&s_ea.mu);
    return NULL;
}

static int vk_ea_on(void)
{
    static int inited;
    if (!inited) {
        inited = 1;
        const char* e = getenv("VK_ENG_ASYNC");
        int n = e && *e ? atoi(e) : 2;
        if (n < 0) n = 0;
        if (n > 8) n = 8;
        pthread_mutex_init(&s_ea.mu, NULL);
        pthread_cond_init(&s_ea.cv, NULL);
        pthread_cond_init(&s_ea.idle, NULL);
        for (int i = 0; i < n; i++)
            if (pthread_create(&s_ea.th[s_ea.n], NULL, vk_ea_main, NULL) == 0) s_ea.n++;
        VK_LOG("engine: %d pipeline compile worker(s)%s\n", s_ea.n, s_ea.n ? "" : " (builds on the FIFO thread)");
    }
    return s_ea.n > 0;
}

static void vk_ea_push(u32 slot)
{
    pthread_mutex_lock(&s_ea.mu);
    if (s_ea.tail - s_ea.head < VK_ENG_JOBQ) s_ea.q[s_ea.tail++ % VK_ENG_JOBQ] = slot;
    pthread_cond_signal(&s_ea.cv);
    pthread_mutex_unlock(&s_ea.mu);
}

/* Ask for topologies of P; queue it when something new is asked. */
static void vk_ea_want(vk_eng_pipe* P, u32 mask)
{
    const u32 was = __atomic_fetch_or(&P->want, mask, __ATOMIC_ACQ_REL);
    if (mask & ~was) vk_ea_push((u32)(P - s_epipe));
}

static void vk_ea_shutdown(void)
{
    if (!s_ea.n) return;
    pthread_mutex_lock(&s_ea.mu);
    s_ea.quit = 1;
    pthread_cond_broadcast(&s_ea.cv);
    pthread_mutex_unlock(&s_ea.mu);
    for (int i = 0; i < s_ea.n; i++) pthread_join(s_ea.th[i], NULL);
    s_ea.n = 0;
}
#else
static int  vk_ea_on(void) { return 0; }
static void vk_ea_want(vk_eng_pipe* P, u32 mask) { (void)P; (void)mask; }
static void vk_ea_shutdown(void) { }
#endif

/* A topology variant of P: built now (synchronous mode) or whatever the
 * workers have finished (asynchronous). */
static VkPipeline vk_eng_variant(vk_eng_pipe* P, u32 topo_idx)
{
    VkPipeline v = __atomic_load_n(&P->variant[topo_idx], __ATOMIC_ACQUIRE);
    if (v || vk_ea_on()) return v;
    v = vk_eng_variant_build(P, topo_idx);
    P->variant[topo_idx] = v;
    P->done |= 1u << topo_idx;
    return v;
}

static void vk_eng_bind_pipeline(void* user, u32 pipeline) { (void)user; s_e2.pipeline = pipeline; }

/* The engine's pipeline catalog: build a pipeline's topology variants now
 * (vk_eng_variant: vkCreateGraphicsPipelines, where MoltenVK translates to MSL
 * and Metal compiles), and report which exist. */
static void vk_eng_pipeline_prewarm(void* user, u32 pipeline, u32 topo_mask)
{
    (void)user;
    if (!pipeline || pipeline > VK_ENG_MAX_PIPE || !s_epipe[pipeline - 1].used) return;
    topo_mask &= (1u << VK_ENG_TOPOLOGIES) - 1u;
    if (vk_ea_on()) { if (topo_mask) vk_ea_want(&s_epipe[pipeline - 1], topo_mask); return; }
    for (u32 t = 0; t < VK_ENG_TOPOLOGIES; t++)
        if ((topo_mask >> t) & 1u) (void)vk_eng_variant(&s_epipe[pipeline - 1], t);
}
static u32 vk_eng_pipeline_variants(void* user, u32 pipeline)
{
    (void)user;
    if (!pipeline || pipeline > VK_ENG_MAX_PIPE || !s_epipe[pipeline - 1].used) return 0;
    u32 m = 0;
    for (u32 t = 0; t < VK_ENG_TOPOLOGIES; t++)
        if (__atomic_load_n(&s_epipe[pipeline - 1].variant[t], __ATOMIC_ACQUIRE)) m |= 1u << t;
    return m;
}

/* Before a draw's textures and vertices are bound: is its variant built? If
 * not, ask the workers for it and have the engine skip the draw. */
static int vk_eng_pipeline_ready(void* user, u32 pipeline, u32 topology)
{
    (void)user;
    if (!vk_ea_on() || !pipeline || pipeline > VK_ENG_MAX_PIPE) return 1;
    vk_eng_pipe* P = &s_epipe[pipeline - 1];
    if (!P->used || topology < 1u || topology > VK_ENG_TOPOLOGIES) return 1;
    if (__atomic_load_n(&P->variant[topology - 1u], __ATOMIC_ACQUIRE)) return 1;
    vk_ea_want(P, 1u << (topology - 1u));
#ifndef _WIN32
    static u64 skipped;
    static double last;
    skipped++;
    const double now = vk_ea_ms();
    if (now - last > 10000.0) {
        last = now;
        pthread_mutex_lock(&s_ea.mu);
        VK_LOG("engine: async compile: %llu variants built, %llu failed, %.0f ms of compiling (max %.1f),"
               " %u queued; %llu draws skipped while waiting\n",
               (unsigned long long)s_ea.built, (unsigned long long)s_ea.failed, s_ea.ms, s_ea.max_ms,
               s_ea.tail - s_ea.head, (unsigned long long)skipped);
        pthread_mutex_unlock(&s_ea.mu);
    }
#endif
    return 0;
}

static void vk_eng2_shutdown(void* user)
{
    vk_ea_shutdown();
    for (u32 i = 1; i <= VK_ENG_MAX_PIPE; i++) vk_eng_pipeline_release(user, i);
    vk_eng_modules_destroy();
    vk_eng_drop_framebuffers();
    for (u32 f = 0; f < VK_ENG_RT_FORMATS; f++)
        for (u32 n = 0; n <= RSX_BE_MAX_COLOR_TARGETS; n++)
            if (s_e2.rp[f][n]) pvkDestroyRenderPass(s_vk.device, s_e2.rp[f][n], NULL);
    vk_destroy_image(&s_e2.scratch_depth);
    vk_destroy_buffer(&s_e2.indices);
    if (s_e2.split_pipe_layout) pvkDestroyPipelineLayout(s_vk.device, s_e2.split_pipe_layout, NULL);
    if (s_e2.split_pool)        pvkDestroyDescriptorPool(s_vk.device, s_e2.split_pool, NULL);
    if (s_e2.split_layout)      pvkDestroyDescriptorSetLayout(s_vk.device, s_e2.split_layout, NULL);
    free(s_e2.ucpu);
    memset(&s_e2, 0, sizeof s_e2);
    for (u32 i = 0; i < s_esamp_count; i++) pvkDestroySampler(s_vk.device, s_esamp[i].s, NULL);
    s_esamp_count = 0;
}

/* Constants are kept in ordinary memory laid out like the shared UBO (vertex
 * at 0, pixel at ps_off) and copied to the GPU per draw: into the draw's ring
 * slice when batched, into the shared UBO otherwise (every such draw is waited
 * on, so overwriting it is safe). They used to be written straight into the
 * UBO, and a batched draw then read them back out of it: that memory is
 * host-coherent, i.e. write-combined and uncached on most drivers, and reading
 * ~8 KB of it took ~40 us -- 85% of a draw's backend time in God of War:
 * Ascension (VK_DRAW_PROF). */
static u8* vk_eng_ucpu(void)
{
    if (!s_e2.ucpu && s_g.ubo.size) s_e2.ucpu = (u8*)calloc(1, (size_t)s_g.ubo.size);
    return s_e2.ucpu;
}
static void vk_eng_bind_vs_constants(void* user, const void* data, u32 bytes)
{
    (void)user;
    if (!s_g.on || !data || !vk_eng_ucpu()) { s_e2.vs_bytes = 0; return; }
    const u32 cap = (u32)s_g.ps_off;
    if (bytes > cap) { bytes = cap; if (!(s_e2.warned & 2u)) { VK_LOG("engine: VS constants truncated\n"); s_e2.warned |= 2u; } }
    memcpy(s_e2.ucpu, data, bytes);
    s_e2.vs_bytes = bytes;
    s_e2.cb_dirty = 1;
}
static void vk_eng_bind_ps_constants(void* user, const void* data, u32 bytes)
{
    (void)user;
    if (!s_g.on || !data || !vk_eng_ucpu()) { s_e2.ps_bytes = 0; return; }
    const u32 cap = (u32)(s_g.ubo.size - s_g.ps_off);
    if (bytes > cap) { bytes = cap; if (!(s_e2.warned & 4u)) { VK_LOG("engine: PS constants truncated\n"); s_e2.warned |= 4u; } }
    memcpy(s_e2.ucpu + s_g.ps_off, data, bytes);
    s_e2.ps_bytes = bytes;
    s_e2.cb_dirty = 1;
}
static void vk_eng_bind_textures(void* user, const u32* textures,
                                 const rsx_be_sampler_desc* samplers, u32 mask)
{
    (void)user;
    for (u32 u = 0; u < RSX_BE_MAX_TEXTURES; u++) {
        const int on = (mask >> u) & 1u;
        s_e2.tex[u] = on ? textures[u] : 0;
        if (on && samplers) s_e2.samp[u] = samplers[u];
    }
    s_e2.tex_mask = mask;
}
static void vk_eng_bind_vertex_textures(void* user, const u32* textures,
                                        const rsx_be_sampler_desc* samplers, u32 mask)
{
    (void)user; (void)textures; (void)samplers;
    if (mask && !(s_e2.warned & 8u)) { VK_LOG("engine: vertex textures not supported yet\n"); s_e2.warned |= 8u; }
}
static void vk_eng_set_viewport(void* user, float x, float y, float w, float h)
{
    (void)user;
    s_e2.vp[0] = x; s_e2.vp[1] = y; s_e2.vp[2] = w; s_e2.vp[3] = h; s_e2.have_vp = 1;
}
static void vk_eng_set_scissor(void* user, u32 x, u32 y, u32 w, u32 h)
{
    (void)user;
    /* VK_SCISSOR_TINY=1 (diagnostic): every draw clipped to 1x1, the ceiling of any resolution cut. */
    { static int tiny = -1; if (tiny < 0) tiny = getenv("VK_SCISSOR_TINY") ? 1 : 0;
      if (tiny) { x = 0; y = 0; w = 1; h = 1; } }
    s_e2.sc[0] = x; s_e2.sc[1] = y; s_e2.sc[2] = w; s_e2.sc[3] = h; s_e2.have_sc = 1;
}
static void vk_eng_set_stencil_ref(void* user, u32 ref) { (void)user; s_e2.stencil_ref = ref; }
static void vk_eng_set_blend_color(void* user, const float rgba[4])
{ (void)user; memcpy(s_e2.blend_color, rgba, sizeof s_e2.blend_color); }

/* Vulkan wants the bounds inside [0, 1] without VK_EXT_depth_range_unrestricted;
 * a NaN clamps to the open end so a garbage register cannot reject a draw. */
static float vk_depth_bound(float v, float open)
{
    if (!(v == v)) return open;
    return v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
}

/* The engine runs the test in the pixel program when this is 0 (MoltenVK). */
static int vk_eng_depth_bounds_native(void* user)
{
    (void)user;
    return s_vk.depth_bounds_ok;
}
static void vk_eng_set_depth_bias(void* user, float slope_factor, float units)
{
    (void)user;
    s_e2.depth_bias[0] = slope_factor;
    s_e2.depth_bias[1] = units;
}

static void vk_eng_set_depth_bounds(void* user, float min, float max)
{
    (void)user;
    s_e2.depth_bounds[0] = vk_depth_bound(min, 0.0f);
    s_e2.depth_bounds[1] = vk_depth_bound(max, 1.0f);
}

static VkFramebuffer vk_eng_framebuffer(VkRenderPass rp, const u32* rt, u32 n, VkImageView dview,
                                        u32 dkey, u32 w, u32 h)
{
    u32 key[RSX_BE_MAX_COLOR_TARGETS + 1] = { 0 };
    for (u32 i = 0; i < n; i++) key[i] = rt[i];
    key[RSX_BE_MAX_COLOR_TARGETS] = dkey;
    for (u32 i = 0; i < s_e2.fb_count; i++)
        if (!memcmp(s_e2.fb[i].key, key, sizeof key)) return s_e2.fb[i].fb;
    if (s_e2.fb_count >= VK_ENG_FB_CACHE) vk_eng_drop_framebuffers();
    VkImageView views[RSX_BE_MAX_COLOR_TARGETS + 1];
    for (u32 i = 0; i < n; i++) views[i] = s_eobj[rt[i]].im.view;
    views[n] = dview;
    VkFramebufferCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                    .renderPass = rp, .attachmentCount = n + 1, .pAttachments = views,
                                    .width = w, .height = h, .layers = 1 };
    VkFramebuffer fb = VK_NULL_HANDLE;
    if (pvkCreateFramebuffer(s_vk.device, &fci, NULL, &fb) != VK_SUCCESS) return VK_NULL_HANDLE;
    memcpy(s_e2.fb[s_e2.fb_count].key, key, sizeof key);
    s_e2.fb[s_e2.fb_count++].fb = fb;
    return fb;
}

/* Depth when the draw names no zeta: one scratch image, grown as needed and
 * cleared to the far plane when (re)created. Key 0xFFFFFFFF in the cache. */
static VkImageView vk_eng_scratch_depth(u32 w, u32 h)
{
    if (s_e2.scratch_depth.view && s_e2.scratch_w >= w && s_e2.scratch_h >= h)
        return s_e2.scratch_depth.view;
    vk_eng_drop_framebuffers();
    vk_destroy_image(&s_e2.scratch_depth);
    if (vk_create_image(&s_e2.scratch_depth, s_vk.depth_format, w, h,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        s_vk.depth_aspect) || vk_begin())
        return VK_NULL_HANDLE;
    vk_barrier_image(s_e2.scratch_depth.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_UNDEFINED);
    VkClearDepthStencilValue far_plane = { 1.0f, 0 };
    VkImageSubresourceRange r = { s_vk.depth_aspect, 0, 1, 0, 1 };
    pvkCmdClearDepthStencilImage(s_vk.cmd, s_e2.scratch_depth.img, VK_IMAGE_LAYOUT_GENERAL, &far_plane, 1, &r);
    if (vk_submit_and_wait()) return VK_NULL_HANDLE;
    s_e2.scratch_w = w; s_e2.scratch_h = h;
    return s_e2.scratch_depth.view;
}

static int vk_eng_ensure_indices(VkDeviceSize bytes)
{
    if (s_e2.indices.buf && s_e2.indices.size >= bytes) return 0;
    VkDeviceSize size = s_e2.indices.size ? s_e2.indices.size : 64 * 1024;
    while (size < bytes) size *= 2;
    vk_destroy_buffer(&s_e2.indices);
    if (vk_create_host_buffer(&s_e2.indices, size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT)) return -1;
    return s_e2.indices.coherent ? 0 : -1;
}

/* rsx_be_sampler_desc is the registers decoded the D3D12 way; the wrap codes
 * stay the guest's 1..8, which vk_gcm_wrap already translates. */
static VkSampler vk_eng_sampler(const rsx_be_sampler_desc* d)
{
    u32 lo, hi;
    memcpy(&lo, &d->min_lod, 4); memcpy(&hi, &d->max_lod, 4);
    const u64 key = (u64)d->min_linear | ((u64)d->mag_linear << 1) | ((u64)d->mip_linear << 2)
                  | ((u64)d->mip_present << 3) | ((u64)(d->wrap_s & 0xF) << 4)
                  | ((u64)(d->wrap_t & 0xF) << 8) | ((u64)(d->wrap_r & 0xF) << 12)
                  | ((u64)(lo >> 16) << 16) | ((u64)(hi >> 16) << 32);
    /* Border colour: opaque white where the guest asks for a white border (a shadow map's far plane),
     * opaque black for an opaque black one, else transparent black. Part of the cache key. */
    VkBorderColor bc = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    if ((d->border_color & 0x00FFFFFFu) == 0x00FFFFFFu) bc = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    else if (d->border_color == 0xFF000000u) bc = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    const u64 key2 = key ^ ((u64)bc << 48) ^ ((u64)(bc != VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK) << 47);
    for (u32 i = 0; i < s_esamp_count; i++) if (s_esamp[i].key == key2) return s_esamp[i].s;
    if (s_esamp_count >= VK_ENG_SAMP_CACHE) return s_vk.sampler;
    /* Each new sampler with a wrap mode past CLAMP_TO_EDGE, once: which scenes use them. */
    if ((d->wrap_s & 0xF) >= 4 || (d->wrap_t & 0xF) >= 4)
        fprintf(stderr, "[vk-sampler] wrap s%u t%u r%u border %08X (sampler %u)\n", d->wrap_s & 0xF,
                d->wrap_t & 0xF, d->wrap_r & 0xF, d->border_color, s_esamp_count);
    float min_lod = d->min_lod, max_lod = d->mip_present ? d->max_lod : d->min_lod;
    if (max_lod < min_lod) max_lod = min_lod;
    VkSamplerCreateInfo sci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = d->mag_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .minFilter = d->min_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .mipmapMode = d->mip_linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = vk_gcm_wrap(d->wrap_s), .addressModeV = vk_gcm_wrap(d->wrap_t),
        .addressModeW = vk_gcm_wrap(d->wrap_r),
        .minLod = min_lod, .maxLod = max_lod,
        .borderColor = bc };
    VkSampler smp;
    if (pvkCreateSampler(s_vk.device, &sci, NULL, &smp) != VK_SUCCESS) return s_vk.sampler;
    s_esamp[s_esamp_count].key = key2;
    s_esamp[s_esamp_count++].s = smp;
    return smp;
}

/* The view a bound engine handle samples through: an uploaded texture, a
 * colour target (bound directly when surface_view was not asked), or a view.
 * A unit gets a view of its own kind only -- a cube for a TextureCube unit, a
 * 2D one otherwise -- and NULL for anything else, which then samples the
 * dummy of the unit's kind. */
static VkImageView vk_eng_sample_view(u32 handle, int want_cube)
{
    if (!handle || handle >= VK_ENG_MAX_OBJ) return VK_NULL_HANDLE;
    const vk_eng_obj* o = &s_eobj[handle];
    if (o->kind != VK_ENG_TEXTURE && o->kind != VK_ENG_COLOR && o->kind != VK_ENG_VIEW)
        return VK_NULL_HANDLE;
    return ((int)o->cube == want_cube) ? o->im.view : VK_NULL_HANDLE;
}

/* What a cube unit with nothing bound samples: a 1x1 cube, magenta on every
 * face like the 2D dummy, made on first use. */
static VkImageView vk_eng_dummy_cube_view(void)
{
    if (!vk_eng_get(s_e2.dummy_cube, VK_ENG_TEXTURE)) {
        static const u8 magenta[4] = { 0xFF, 0x00, 0xFF, 0xFF };
        s_e2.dummy_cube = vk_eng_texture_create(NULL, RSX_BE_FMT_R8G8B8A8, 1, 1, 1, 6,
                                                0xAAE4u, 0x85u);
        if (!s_e2.dummy_cube) return VK_NULL_HANDLE;
        for (u32 f = 0; f < 6; f++)
            vk_eng_texture_upload(NULL, s_e2.dummy_cube, f, 0, 1, 1, magenta, 4, 1);
    }
    return s_eobj[s_e2.dummy_cube].im.view;
}

/* The same for a 3D unit: a 1x1x1 volume. */
static VkImageView vk_eng_dummy_vol_view(void)
{
    if (!vk_eng_get(s_e2.dummy_vol, VK_ENG_TEXTURE)) {
        static const u8 magenta[4] = { 0xFF, 0x00, 0xFF, 0xFF };
        s_e2.dummy_vol = vk_eng_texture_create_3d(NULL, RSX_BE_FMT_R8G8B8A8, 1, 1, 1, 1, 0xAAE4u, 0x85u);
        if (!s_e2.dummy_vol) return VK_NULL_HANDLE;
        vk_eng_texture_upload_3d(NULL, s_e2.dummy_vol, 0, 0, 1, 1, magenta, 4);
    }
    return s_eobj[s_e2.dummy_vol].im.view;
}

/* ---------------------------------------------------------------------------
 * Draw batching (VK_DRAW_BATCH=1)
 *
 * Every vk_eng_draw used to submit its own command buffer and wait on a
 * fence, because the vertex, index and uniform buffers and the descriptor set
 * are single objects rewritten per draw. In God of War: Ascension's first level
 * that was 54,759 draws in 20 s at ~168 us each: the FIFO walker spent 46% of
 * its time blocked on the GPU, the RSX ring drained one draw at a time and the
 * SPU geometry jobs waited for ring space. Batched, a draw takes slices of
 * three ring buffers and a descriptor set from a pool, records into the open
 * command buffer, and only s_eb_pending grows. vk_eng_flush() submits, waits
 * once, and resets the rings and the pool; vk_begin, every object destroy, the
 * present and the engine's own submit point call it first, so all other work
 * keeps seeing everything before it complete. Between batched draws a full
 * memory barrier keeps render-to-texture ordering without tracking.
 * VK_DRAW_BATCH_MAX (default 512) caps the draws per submission.
 *
 * VK_ASYNC=1: a flush submits without waiting. Each of VK_EB_SLOTS batches in
 * flight has its own command buffer, fence, descriptor pool and a slice of
 * every ring; a slot is waited on only when it comes round again. The GPU then
 * runs a batch while the walker records the next, and a readback (which still
 * waits) finds most of the work done. Destroys and the engine's submit point
 * wait for every batch in flight (vk_eng_idle), so objects are never freed
 * under the GPU. Default VK_DRAW_BATCH_MAX with it: 128.
 * -------------------------------------------------------------------------*/
#define VK_EB_SLOTS      3u
#define VK_EB_VB_BYTES   (96u << 20)   /* each ring is split evenly between the slots */
#define VK_EB_IB_BYTES   (24u << 20)
#define VK_EB_UB_BYTES   (48u << 20)
#define VK_EB_SB_BYTES   (96u << 20)   /* texture upload staging */
#define VK_EB_RB_BYTES   (96u << 20)   /* VK_ENG_VTX_GPU: raw guest vertex bytes and decode headers */
#define VK_EB_MAX_SETS   1024u

static struct {
    int              on;            /* -1 undecided, 0 off, 1 on */
    int              async;         /* VK_ASYNC: flushes do not wait */
    u32              max_draws;
    vk_buffer        vb, ib, ub, sb, rb;
    VkDeviceSize     vb_head, ib_head, ub_head, sb_head, rb_head, ub_align;
    VkDeviceSize     vb_cap, ib_cap, ub_cap, sb_cap, rb_cap;   /* end of the current slot's slice */
    VkDescriptorPool pool;          /* the current slot's */
    u32              sets;
    u32              cur;           /* current slot */
    struct { VkCommandBuffer cmd, pro; VkFence fence; VkDescriptorPool pool; u32 sets; int busy, pro_open;
             uint64_t seq; /* vk_queue_submit's count after its last submission (vk_grave_done) */ } slot[VK_EB_SLOTS];
    unsigned long long draws, flushes, slot_waits;
} s_eb = { .on = -1 };

/* Make slot i current: wait for its last batch if it is still in flight, then
 * hand its command buffer, fence, pool and ring slices to the recorder. */
static u32 s_eb_pool_gen;      /* bumped whenever the current descriptor pool changes or is reset */
static void vk_eb_use_slot(u32 i)
{
    if (s_eb.slot[i].busy) {
        const uint64_t ft_t0 = PS3_FT_ON() ? ps3_ft_now() : 0;
        g_vk_fence_waiting = 1;
        pvkWaitForFences(s_vk.device, 1, &s_eb.slot[i].fence, VK_TRUE, UINT64_MAX);
        g_vk_fence_waiting = 0;
        if (ft_t0) {
            const uint64_t ft_t1 = ps3_ft_now();
            if (ft_t1 - ft_t0 >= ps3_ft_min_ns()) ps3_ft_span(PS3_FT_GPU_WAIT, 0, 4, 0, ft_t0, ft_t1);
        }
        s_eb.slot[i].busy = 0;
        s_eb.slot_waits++;
    }
    /* This slot's last batch has completed, and with it everything before it. */
    if (s_grave_n) vk_grave_done(s_eb.slot[i].seq);
    if (s_eb.slot[i].sets) { pvkResetDescriptorPool(s_vk.device, s_eb.slot[i].pool, 0); s_eb.slot[i].sets = 0; }
    s_eb.cur = i;
    s_vk.cmd = s_eb.slot[i].cmd;
    s_vk.fence = s_eb.slot[i].fence;
    s_eb.pool = s_eb.slot[i].pool;
    s_eb.sets = 0;
    s_eb_pool_gen++;
    const u32 n = s_eb.async ? VK_EB_SLOTS : 1u;
    s_eb.vb_head = (VkDeviceSize)i * (VK_EB_VB_BYTES / n); s_eb.vb_cap = s_eb.vb_head + VK_EB_VB_BYTES / n;
    s_eb.ib_head = (VkDeviceSize)i * (VK_EB_IB_BYTES / n); s_eb.ib_cap = s_eb.ib_head + VK_EB_IB_BYTES / n;
    s_eb.ub_head = (VkDeviceSize)i * (VK_EB_UB_BYTES / n); s_eb.ub_cap = s_eb.ub_head + VK_EB_UB_BYTES / n;
    s_eb.sb_head = (VkDeviceSize)i * (VK_EB_SB_BYTES / n); s_eb.sb_cap = s_eb.sb_head + VK_EB_SB_BYTES / n;
    s_eb.rb_head = (VkDeviceSize)i * (VK_EB_RB_BYTES / n); s_eb.rb_cap = s_eb.rb_head + VK_EB_RB_BYTES / n;
    s_eb.slot[i].pro_open = 0;
}

/* A synchronous submission has completed, and with it everything queued before. */
static void vk_eb_all_done(void)
{
    for (u32 i = 0; i < VK_EB_SLOTS; i++) s_eb.slot[i].busy = 0;
}

/* Submit the open batch and wait for every batch in flight: before freeing an
 * object the GPU may still use, and wherever the engine needs results. */
static void vk_eng_idle(void)
{
    if (s_eb_pending) vk_eng_flush();
    if (!s_eb.async) return;
    for (u32 i = 0; i < VK_EB_SLOTS; i++)
        if (s_eb.slot[i].busy) {
            g_vk_fence_waiting = 1;
            pvkWaitForFences(s_vk.device, 1, &s_eb.slot[i].fence, VK_TRUE, UINT64_MAX);
            g_vk_fence_waiting = 0;
            s_eb.slot[i].busy = 0;
        }
    /* Every engine batch has completed: nothing queued for destruction is still in use. */
    if (s_grave_n) vk_grave_done(~(uint64_t)0);
}

static int vk_eb_on(void)
{
    if (s_eb.on >= 0) return s_eb.on;
    const char* e = getenv("VK_DRAW_BATCH");
    s_eb.on = 0;
    if (!e || *e != '1' || !s_g.on) return 0;
    const char* as = getenv("VK_ASYNC");
    s_eb.async = as && *as == '1';
    const char* m = getenv("VK_DRAW_BATCH_MAX");
    s_eb.max_draws = m && atoi(m) > 0 ? (u32)atoi(m) : s_eb.async ? 128u : 512u;
    if (s_eb.max_draws > VK_EB_MAX_SETS) s_eb.max_draws = VK_EB_MAX_SETS;
    VkPhysicalDeviceProperties p;
    pvkGetPhysicalDeviceProperties(s_vk.phys, &p);
    s_eb.ub_align = p.limits.minUniformBufferOffsetAlignment ? p.limits.minUniformBufferOffsetAlignment : 256;
    VkDescriptorPoolSize ps[3] = {
        { VK_UB_TYPE, 2 * VK_EB_MAX_SETS },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, RSX_MAX_TEXTURES * VK_EB_MAX_SETS },
        { VK_DESCRIPTOR_TYPE_SAMPLER, RSX_MAX_TEXTURES * VK_EB_MAX_SETS },
    };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = VK_EB_MAX_SETS, .poolSizeCount = 3, .pPoolSizes = ps };
    if (vk_create_host_buffer(&s_eb.vb, VK_EB_VB_BYTES, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) || !s_eb.vb.coherent ||
        vk_create_host_buffer(&s_eb.rb, VK_EB_RB_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) || !s_eb.rb.coherent ||
        vk_create_host_buffer(&s_eb.ib, VK_EB_IB_BYTES, VK_BUFFER_USAGE_INDEX_BUFFER_BIT) || !s_eb.ib.coherent ||
        vk_create_host_buffer(&s_eb.ub, VK_EB_UB_BYTES, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) || !s_eb.ub.coherent ||
        vk_create_host_buffer(&s_eb.sb, VK_EB_SB_BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT) || !s_eb.sb.coherent) {
        VK_LOG("draw batching: no host-coherent ring memory, staying off\n");
        return 0;
    }
    /* Slot 0 is the recorder's own command buffer and fence; the others are
     * only made for VK_ASYNC. */
    const u32 nslots = s_eb.async ? VK_EB_SLOTS : 1u;
    s_eb.slot[0].cmd = s_vk.cmd;
    s_eb.slot[0].fence = s_vk.fence;
    for (u32 i = 0; i < nslots; i++) {
        if (pvkCreateDescriptorPool(s_vk.device, &pci, NULL, &s_eb.slot[i].pool) != VK_SUCCESS) {
            VK_LOG("draw batching: no descriptor pool, staying off\n");
            return 0;
        }
        {   /* VK_ENG_VTX_GPU: a prologue submitted ahead of the slot's batch */
            VkCommandBufferAllocateInfo pai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                                .commandPool = s_vk.pool,
                                                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                .commandBufferCount = 1 };
            if (pvkAllocateCommandBuffers(s_vk.device, &pai, &s_eb.slot[i].pro) != VK_SUCCESS)
                s_eb.slot[i].pro = VK_NULL_HANDLE;
        }
        if (i == 0) continue;
        VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                            .commandPool = s_vk.pool,
                                            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                            .commandBufferCount = 1 };
        VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        if (pvkAllocateCommandBuffers(s_vk.device, &cai, &s_eb.slot[i].cmd) != VK_SUCCESS ||
            pvkCreateFence(s_vk.device, &fci, NULL, &s_eb.slot[i].fence) != VK_SUCCESS) {
            VK_LOG("draw batching: no command buffer for an in-flight slot, async off\n");
            s_eb.async = 0;
            break;
        }
    }
    vk_eb_use_slot(0);
    s_eb_async = s_eb.async;
    VK_LOG("draw batching: on (up to %u draws per submission%s)\n", s_eb.max_draws,
           s_eb.async ? ", 3 submissions in flight" : "");
    s_eb.on = 1;
    return 1;
}

static int vk_eb_active(void) { return s_eb.on == 1; }
static u32 vk_eb_max(void) { return s_eb.max_draws; }
static void vk_eb_recorded(void) { if (++s_eb_pending >= s_eb.max_draws) vk_eng_flush(); }

static VkDeviceSize vk_eb_take(VkDeviceSize* head, VkDeviceSize cap, VkDeviceSize bytes, VkDeviceSize align);

/* A texture upload as one more batched operation: rows staged in the ring, the
 * copy recorded behind a full memory barrier (earlier draws of the batch may
 * still sample the old texels) and followed by one on the image (later draws
 * sample the new ones). Each used to allocate a host buffer and wait on its own
 * submission, ~3000 times in 20 s of God of War: Ascension's first level. */
static int vk_eb_upload(VkImage img, VkBufferImageCopy* region, const void* src, size_t tight,
                        u32 rows, u32 row_bytes)
{
    if (!vk_eb_active()) return -2;
    const VkDeviceSize bytes = (VkDeviceSize)tight * rows;
    if (!bytes) return -2;
    VkDeviceSize off = vk_eb_take(&s_eb.sb_head, s_eb.sb_cap, bytes, 16u);
    if (off == (VkDeviceSize)-1) {
        vk_eng_flush();
        off = vk_eb_take(&s_eb.sb_head, s_eb.sb_cap, bytes, 16u);
        if (off == (VkDeviceSize)-1) return -2;       /* larger than a slot's slice */
    }
    for (u32 y = 0; y < rows; y++)
        memcpy((u8*)s_eb.sb.ptr + off + (size_t)y * tight, (const u8*)src + (size_t)y * row_bytes, tight);
    if (!s_eb_pending && vk_begin()) return -1;
    vk_cmd_after_everything(s_vk.cmd);
    region->bufferOffset = off;
    pvkCmdCopyBufferToImage(s_vk.cmd, s_eb.sb.buf, img, VK_IMAGE_LAYOUT_GENERAL, 1, region);
    vk_barrier_image(img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
    if (++s_eb_pending >= s_eb.max_draws) vk_eng_flush();
    return 0;
}

static void vk_eng_flush(void)
{
    if (!s_eb_pending) return;
    vk_pass_end();
    s_eb_pending = 0;               /* first: the submit below must not re-enter */
    s_eb.slot[s_eb.cur].sets = s_eb.sets;
    if (s_eb.async) {
        if (pvkEndCommandBuffer(s_vk.cmd) == VK_SUCCESS &&
            pvkResetFences(s_vk.device, 1, &s_vk.fence) == VK_SUCCESS) {
            /* VK_ENG_VTX_GPU: the vertex decodes go first, and their writes are
             * visible to every vertex fetch of the batch after them. */
            VkCommandBuffer cbs[2];
            u32 ncb = 0;
            if (s_eb.slot[s_eb.cur].pro_open) {
                VkCommandBuffer pro = s_eb.slot[s_eb.cur].pro;
                VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                       .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                       .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT };
                pvkCmdPipelineBarrier(pro, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                                      0, 1, &mb, 0, NULL, 0, NULL);
                if (pvkEndCommandBuffer(pro) == VK_SUCCESS) cbs[ncb++] = pro;
                s_eb.slot[s_eb.cur].pro_open = 0;
            }
            cbs[ncb++] = s_vk.cmd;
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                .commandBufferCount = ncb, .pCommandBuffers = cbs };
            if (vk_queue_submit(&si, s_vk.fence) == VK_SUCCESS) {
                s_eb.slot[s_eb.cur].busy = 1;
                s_eb.slot[s_eb.cur].seq = s_vk_sub_seq;
            } else VK_LOG("draw batching: vkQueueSubmit failed\n");
        }
        vk_eb_use_slot((s_eb.cur + 1u) % VK_EB_SLOTS);
    } else {
        vk_submit_and_wait();
        vk_eb_use_slot(0);
    }
    if ((++s_eb.flushes & 0x3FFFu) == 0)
        VK_LOG("draw batching: %llu draws in %llu submissions, %llu waits for a slot\n",
               s_eb.draws, s_eb.flushes, s_eb.slot_waits);
}

static VkDeviceSize vk_eb_take(VkDeviceSize* head, VkDeviceSize cap, VkDeviceSize bytes, VkDeviceSize align)
{
    const VkDeviceSize at = (*head + align - 1u) / align * align;
    if (at + bytes > cap) return (VkDeviceSize)-1;
    *head = at + bytes;
    return at;
}

static VkDescriptorSet vk_eng_write_descriptors_to(const vk_eng_pipe* P, VkDescriptorSet set,
                                                   VkBuffer ubo, VkDeviceSize vs_off, VkDeviceSize ps_off);
/* VK_DRAW_PROF's split of the descriptor work: set allocation, uniform
 * copies, view and sampler resolve, vkUpdateDescriptorSets. */
static unsigned long long s_dq_us[4];
static unsigned long long s_dl_reused;   /* VK_UB_DYNAMIC: draws that bound the previous set again */
static unsigned long long s_cb_reused;   /* VK_CB_REUSE: ...and its uniform slices too */
static unsigned long long vk_dp_now(void);

/* The views and samplers a draw's set would hold (VK_UB_DYNAMIC's reuse key). */
static void vk_eng_resolve_bindings(const vk_eng_pipe* P, VkImageView* v, VkSampler* s)
{
    for (u32 u = 0; u < RSX_MAX_TEXTURES; u++) {
        const int cube = ((P->vol_mask >> u) & 1u) ? 2 : (int)((P->cube_mask >> u) & 1u);
        v[u] = ((s_e2.tex_mask >> u) & 1u) ? vk_eng_sample_view(s_e2.tex[u], cube) : VK_NULL_HANDLE;
        s[u] = v[u] ? vk_eng_sampler(&s_e2.samp[u]) : s_vk.sampler;
    }
}

/* Fill the set the bound pipeline uses: the shared one (sixteen textures in
 * one binding), or for a pipeline with cube units the split one (one binding
 * per unit). Returns the set, or VK_NULL_HANDLE. */
static VkDescriptorSet vk_eng_write_descriptors(const vk_eng_pipe* P)
{
    const VkDescriptorSet set = P->cube_mask ? s_e2.split_set : s_g.set;
    return vk_eng_write_descriptors_to(P, set, s_g.ubo.buf, 0, s_g.ps_off);
}

static VkDescriptorSet vk_eng_write_descriptors_to(const vk_eng_pipe* P, VkDescriptorSet set,
                                                   VkBuffer ubo, VkDeviceSize vs_off, VkDeviceSize ps_off)
{
    const int split = P->cube_mask != 0;
    if (!set) return VK_NULL_HANDLE;
    const VkImageView dummy_cube = split ? vk_eng_dummy_cube_view() : VK_NULL_HANDLE;
    if (split && !dummy_cube) return VK_NULL_HANDLE;
    const VkImageView dummy_vol = P->vol_mask ? vk_eng_dummy_vol_view() : VK_NULL_HANDLE;
    if (P->vol_mask && !dummy_vol) return VK_NULL_HANDLE;
    VkDescriptorBufferInfo bi[2] = {
        { ubo, vk_ub_dynamic() ? 0 : vs_off, s_e2.vs_bytes ? s_e2.vs_bytes : 16u },
        { ubo, vk_ub_dynamic() ? 0 : ps_off, s_e2.ps_bytes ? s_e2.ps_bytes : 16u },
    };
    VkDescriptorImageInfo ii[RSX_MAX_TEXTURES], si[RSX_MAX_TEXTURES];
    const unsigned long long q0 = vk_dp_now();
    for (u32 u = 0; u < RSX_MAX_TEXTURES; u++) {
        const int cube = ((P->vol_mask >> u) & 1u) ? 2 : (int)((P->cube_mask >> u) & 1u);
        VkImageView v = ((s_e2.tex_mask >> u) & 1u) ? vk_eng_sample_view(s_e2.tex[u], cube)
                                                     : VK_NULL_HANDLE;
        VkSampler smp = v ? vk_eng_sampler(&s_e2.samp[u]) : s_vk.sampler;
        ii[u] = (VkDescriptorImageInfo){ VK_NULL_HANDLE,
                                         v ? v : (cube == 2 ? dummy_vol : cube ? dummy_cube : s_vk.dummy.view),
                                         VK_IMAGE_LAYOUT_GENERAL };
        si[u] = (VkDescriptorImageInfo){ smp, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
    }
    VkWriteDescriptorSet w[3 + RSX_MAX_TEXTURES];
    u32 n = 0;
    w[n++] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
          .dstBinding = RSX_SPIRV_VPCONST_BINDING, .descriptorCount = 1,
          .descriptorType = VK_UB_TYPE, .pBufferInfo = &bi[0] };
    w[n++] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
          .dstBinding = RSX_SPIRV_PSCONST_BINDING, .descriptorCount = 1,
          .descriptorType = VK_UB_TYPE, .pBufferInfo = &bi[1] };
    w[n++] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
          .dstBinding = RSX_SPIRV_SAMPLER_BINDING, .descriptorCount = RSX_MAX_TEXTURES,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .pImageInfo = si };
    if (split) {
        for (u32 u = 0; u < RSX_MAX_TEXTURES; u++)
            w[n++] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                  .dstSet = set, .dstBinding = RSX_SPIRV_TEXTURE_BINDING + u, .descriptorCount = 1,
                  .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &ii[u] };
    } else {
        w[n++] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
              .dstBinding = RSX_SPIRV_TEXTURE_BINDING, .descriptorCount = RSX_MAX_TEXTURES,
              .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = ii };
    }
    const unsigned long long q1 = q0 ? vk_dp_now() : 0;
    pvkUpdateDescriptorSets(s_vk.device, n, w, 0, NULL);
    if (q0) { s_dq_us[2] += q1 - q0; s_dq_us[3] += vk_dp_now() - q1; }
    return set;
}

/* VK_DRAW_PROF=<s>: every <s> seconds, where vk_eng_draw's time goes -- cache
 * lookups (pipeline variant, render pass, framebuffer), descriptor set and
 * uniform slices, vertex/index copies, command recording, and the batch flush
 * (submit, and the wait for a free in-flight slot). */
static int s_dp_s = -1;
static unsigned long long s_dp_w0, s_dp_us[5], s_dp_draws;
static const char* const s_dp_name[5] = { "lookup", "desc", "copy", "record", "flush" };
static unsigned long long vk_dp_now(void)
{
    if (s_dp_s < 0) { const char* e = getenv("VK_DRAW_PROF"); s_dp_s = e ? atoi(e) : 0; }
    if (s_dp_s <= 0) return 0;
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (unsigned long long)ts.tv_sec * 1000000ull + (unsigned long long)ts.tv_nsec / 1000ull;
}
static void vk_dp_end(const unsigned long long t[6])
{
    if (!t[0]) return;
    for (int i = 0; i < 5; i++) s_dp_us[i] += t[i + 1] - t[i];
    s_dp_draws++;
    if (!s_dp_w0) s_dp_w0 = t[0];
    if (t[5] - s_dp_w0 < (unsigned long long)s_dp_s * 1000000ull) return;
    fprintf(stderr, "[vk-draw-prof] %.0f ms: %llu draws, us/draw:", (t[5] - s_dp_w0) / 1000.0, s_dp_draws);
    for (int i = 0; i < 5; i++)
        fprintf(stderr, " %s %.1f", s_dp_name[i], (double)s_dp_us[i] / (double)s_dp_draws);
    fprintf(stderr, " | ms: %.0f %.0f %.0f %.0f %.0f | desc us/draw: alloc %.1f ubcopy %.1f resolve %.1f update %.1f"
            " (ub %u+%u B)\n", s_dp_us[0] / 1000.0, s_dp_us[1] / 1000.0,
            s_dp_us[2] / 1000.0, s_dp_us[3] / 1000.0, s_dp_us[4] / 1000.0,
            (double)s_dq_us[0] / s_dp_draws, (double)s_dq_us[1] / s_dp_draws,
            (double)s_dq_us[2] / s_dp_draws, (double)s_dq_us[3] / s_dp_draws, s_e2.vs_bytes, s_e2.ps_bytes);
    fprintf(stderr, "[vk-draw-prof]   passes %llu, draws into an open pass %llu, new pass for a sampled target %llu, sets reused %llu (uniform slices too %llu)\n",
            s_pass_n[0], s_pass_n[1], s_pass_n[2], s_dl_reused, s_cb_reused);
    memset(s_pass_n, 0, sizeof s_pass_n);
    memset(s_dp_us, 0, sizeof s_dp_us);
    memset(s_dq_us, 0, sizeof s_dq_us);
    s_dp_draws = 0;
    s_dp_w0 = t[5];
}

/* VK_ENG_VTX_GPU (the engine's RSX_VTX_GPU): a draw's guest vertex bytes are
 * copied raw into the batch's raw ring and decoded by a compute pass
 * (vulkan/rsx_vk_vtx_decode.comp) into its vertex ring slice. The dispatches
 * go into the slot's prologue command buffer, submitted ahead of the batch, so
 * render passes stay open across draws. The formats the CPU decodes with a
 * division (UNORM8, SNORM16) or the half rules use tables filled by
 * rsx_vertex_decode_element itself, so every value is the CPU's bits. */
static struct {
    int tried, ok;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    vk_buffer tab;
} s_vd;

static int vk_vd_init(void)
{
    if (s_vd.tried) return s_vd.ok;
    s_vd.tried = 1;
    if (!s_eb.rb.buf || !s_eb.async) return 0;
    for (u32 i = 0; i < VK_EB_SLOTS; i++) if (!s_eb.slot[i].pro) return 0;
    const u32 nt = 256u + 65536u + 65536u;
    if (vk_create_host_buffer(&s_vd.tab, (VkDeviceSize)nt * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) || !s_vd.tab.coherent)
        return 0;
    u32* tw = (u32*)s_vd.tab.ptr;
    for (u32 x = 0; x < nt; x++) {
        const u32 type = x < 256u ? RSX_VTX_TYPE_UNORM8 : x < 65792u ? RSX_VTX_TYPE_SNORM16 : RSX_VTX_TYPE_HALF;
        const u32 k = x < 256u ? x : x < 65792u ? x - 256u : x - 65792u;
        const u8 b[2] = { (u8)(type == RSX_VTX_TYPE_UNORM8 ? k : k >> 8), (u8)k };
        float f[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
        rsx_vertex_decode_element(type, 1, b, f);
        memcpy(&tw[x], &f[0], 4);
    }
    VkDescriptorSetLayoutBinding b[3];
    for (u32 i = 0; i < 3; i++)
        b[i] = (VkDescriptorSetLayoutBinding){ .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                               .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .bindingCount = 3, .pBindings = b };
    VkPushConstantRange pr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 };
    VkPipelineLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                       .setLayoutCount = 1, .pSetLayouts = &s_vd.dsl,
                                       .pushConstantRangeCount = 1, .pPushConstantRanges = &pr };
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
    VkShaderModule sm = VK_NULL_HANDLE;
    if (pvkCreateDescriptorSetLayout(s_vk.device, &dci, NULL, &s_vd.dsl) != VK_SUCCESS ||
        pvkCreatePipelineLayout(s_vk.device, &lci, NULL, &s_vd.layout) != VK_SUCCESS ||
        pvkCreateDescriptorPool(s_vk.device, &pci, NULL, &s_vd.pool) != VK_SUCCESS ||
        vk_create_shader(rsx_vk_vtx_decode_comp_spv, sizeof rsx_vk_vtx_decode_comp_spv, &sm)) {
        VK_LOG("engine: vertex decode compute setup failed\n");
        return 0;
    }
    VkComputePipelineCreateInfo cci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" },
        .layout = s_vd.layout };
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = s_vd.pool, .descriptorSetCount = 1,
                                        .pSetLayouts = &s_vd.dsl };
    int ok = pvkCreateComputePipelines(s_vk.device, VK_NULL_HANDLE, 1, &cci, NULL, &s_vd.pipe) == VK_SUCCESS &&
             pvkAllocateDescriptorSets(s_vk.device, &dai, &s_vd.set) == VK_SUCCESS;
    pvkDestroyShaderModule(s_vk.device, sm, NULL);
    if (ok) {
        VkDescriptorBufferInfo bi[3] = { { s_eb.rb.buf, 0, VK_WHOLE_SIZE }, { s_eb.vb.buf, 0, VK_WHOLE_SIZE },
                                         { s_vd.tab.buf, 0, VK_WHOLE_SIZE } };
        VkWriteDescriptorSet wr[3];
        for (u32 i = 0; i < 3; i++)
            wr[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s_vd.set,
                                            .dstBinding = i, .descriptorCount = 1,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            .pBufferInfo = &bi[i] };
        pvkUpdateDescriptorSets(s_vk.device, 3, wr, 0, NULL);
    } else {
        VK_LOG("engine: vertex decode compute pipeline failed\n");
    }
    s_vd.ok = ok;
    if (ok) VK_LOG("engine: vertex decode on the GPU ready\n");
    return ok;
}

/* Interleaved attributes share their guest bytes: slots with one stride whose
 * first elements lie within one stride of each other form a group, copied
 * once (its union), and each slot reads at its own offset into it. */
static u32 s_vd_ng, s_vd_grp[16];
static const u8* s_vd_g0[16];
static const u8* s_vd_g1[16];
static u32 s_vd_gs[16];

static void vk_vd_group(const rsx_be_vtx_job* job)
{
    s_vd_ng = 0;
    for (u32 s = 0; s < job->nslots; s++) {
        const rsx_be_vtx_slot* js = &job->slot[s];
        if (!js->type) continue;
        u32 g = 0;
        while (g < s_vd_ng && !(s_vd_gs[g] == js->stride && js->stride &&
                                js->src + js->stride > s_vd_g0[g] && js->src < s_vd_g0[g] + js->stride)) g++;
        if (g == s_vd_ng) { s_vd_ng++; s_vd_g0[g] = js->src; s_vd_g1[g] = js->src + js->bytes; s_vd_gs[g] = js->stride; }
        if (js->src < s_vd_g0[g]) s_vd_g0[g] = js->src;
        if (js->src + js->bytes > s_vd_g1[g]) s_vd_g1[g] = js->src + js->bytes;
        s_vd_grp[s] = g;
    }
}

/* Raw-ring bytes a job needs: a header of 4 + 8 words per slot, then each
 * group's guest bytes, 4-aligned (vk_vd_group first). */
static VkDeviceSize vk_vd_bytes(const rsx_be_vtx_job* job)
{
    VkDeviceSize b = (4u + 8u * job->nslots) * 4u;
    for (u32 g = 0; g < s_vd_ng; g++) b += ((VkDeviceSize)(s_vd_g1[g] - s_vd_g0[g]) + 3u) & ~(VkDeviceSize)3u;
    return b;
}

static struct { unsigned long long draws, checked, differ; time_t t; } s_vds;

static int vk_eng_draw_impl(rsx_topology topology, const void* vertices, u32 vertex_count,
                            u32 stride, const u32* indices, u32 index_count,
                            const rsx_be_vtx_job* job, const void* check)
{
    unsigned long long dp[6];
    dp[0] = vk_dp_now();
    if (job) {
        if (!vk_eb_on() || !s_eb.async || !vk_vd_init() || !job->count || !job->nslots ||
            job->nslots * 16u != stride)
            return -1;
        vertex_count = job->count;
        vk_vd_group(job);
    }
    if (!s_e2.pipeline || (!vertices && !job) || !vertex_count || !stride) return -1;
    if (topology < RSX_TOPOLOGY_POINTS || topology > RSX_TOPOLOGY_TRIANGLE_STRIP) return -1;
    vk_eng_pipe* P = &s_epipe[s_e2.pipeline - 1];
    if (!P->used || stride != P->stride) return -1;

    /* Targets: every member of a set is target A's size (engine contract). */
    const u32 n = s_eng_bound.nrt;
    if (n < 1 || n != P->rt_count) {
        if (!(s_e2.warned & 16u)) { VK_LOG("engine: draw with %u colour targets for a %u-target pipeline skipped\n", n, P->rt_count); s_e2.warned |= 16u; }
        return -1;
    }
    vk_eng_obj* rt0 = vk_eng_get(s_eng_bound.rt[0], VK_ENG_COLOR);
    if (!rt0) return -1;
    for (u32 i = 1; i < n; i++)
        if (!vk_eng_get(s_eng_bound.rt[i], VK_ENG_COLOR)) return -1;
    if (rt0->fmt != P->rt_fmt) {
        if (!(s_e2.warned & 32u)) { VK_LOG("engine: draw into colour format %d with a pipeline built for %d skipped\n", (int)rt0->fmt, (int)P->rt_fmt); s_e2.warned |= 32u; }
        return -1;
    }
    const u32 w = rt0->w, h = rt0->h;
    vk_eng_obj* zo = vk_eng_get(s_eng_bound.depth, VK_ENG_DEPTH);
    VkImageView dview = (zo && zo->w >= w && zo->h >= h) ? zo->im.view : vk_eng_scratch_depth(w, h);
    if (!dview) return -1;
    const u32 dkey = (zo && dview == zo->im.view) ? s_eng_bound.depth : 0xFFFFFFFFu;

    VkPipeline pipe = vk_eng_variant(P, (u32)topology - 1u);
    VkRenderPass rp = vk_eng_render_pass(P->rt_fmt, n);
    VkFramebuffer fb = vk_eng_framebuffer(rp, s_eng_bound.rt, n, dview, dkey, w, h);
    if (!pipe || !rp || !fb) return -1;
    dp[1] = dp[0] ? vk_dp_now() : 0;

    const VkDeviceSize vbytes = (VkDeviceSize)vertex_count * stride;
    const VkDeviceSize ibytes = (indices && index_count) ? (VkDeviceSize)index_count * 4u : 0;
    VkBuffer vbuf = s_vk.vertices.buf, ibuf = s_e2.indices.buf;
    VkDeviceSize voff = 0, ioff = 0, roff = 0;
    VkDescriptorSet set = VK_NULL_HANDLE;
    uint32_t dyn_off[2] = { 0, 0 };   /* VK_UB_DYNAMIC: the uniform slices, bound with the set */
    const u32 vsb = s_e2.vs_bytes ? s_e2.vs_bytes : 16u, psb = s_e2.ps_bytes ? s_e2.ps_bytes : 16u;
    int batched = vk_eb_on() && vbytes <= VK_EB_VB_BYTES / VK_EB_SLOTS && ibytes <= VK_EB_IB_BYTES / VK_EB_SLOTS;
    if (batched && P->cube_mask && !vk_eng_dummy_cube_view()) return -1;
    if (batched && P->vol_mask && !vk_eng_dummy_vol_view()) return -1;   /* may flush: before any slice */
    if (batched) {
        /* Slices of the rings and a set from the pool; a full one flushes the
         * batch (which empties them all) and the draw takes its slices again. */
        for (int tries = 0; tries < 2; tries++) {
            VkDeviceSize vh = s_eb.vb_head, ih = s_eb.ib_head, uh = s_eb.ub_head, rh = s_eb.rb_head;
            voff = vk_eb_take(&vh, s_eb.vb_cap, vbytes, 16u);
            if (job && voff != (VkDeviceSize)-1) {
                roff = vk_eb_take(&rh, s_eb.rb_cap, vk_vd_bytes(job), 16u);
                if (roff == (VkDeviceSize)-1) voff = (VkDeviceSize)-1;
            }
            ioff = ibytes ? vk_eb_take(&ih, s_eb.ib_cap, ibytes, 16u) : 0;
            const VkDeviceSize uvs = vk_eb_take(&uh, s_eb.ub_cap, vsb, s_eb.ub_align);
            const VkDeviceSize ups = uvs == (VkDeviceSize)-1 ? uvs : vk_eb_take(&uh, s_eb.ub_cap, psb, s_eb.ub_align);
            if (voff != (VkDeviceSize)-1 && ioff != (VkDeviceSize)-1 && ups != (VkDeviceSize)-1 &&
                s_eb.sets < VK_EB_MAX_SETS) {
                /* VK_UB_DYNAMIC: the previous draw's set, when it came from this pool
                 * and holds the same views and samplers; only the uniform slices move. */
                static struct { VkDescriptorPool pool; u32 gen, cube; VkDescriptorSet set; VkBuffer ubo;
                                VkImageView v[RSX_MAX_TEXTURES]; VkSampler s[RSX_MAX_TEXTURES];
                                VkDeviceSize uvs, ups; u32 vsb, psb; } s_dl;
                VkImageView rv[RSX_MAX_TEXTURES]; VkSampler rs[RSX_MAX_TEXTURES];
                if (vk_ub_dynamic()) {
                    vk_eng_resolve_bindings(P, rv, rs);
                    if (s_dl.set && s_dl.pool == s_eb.pool && s_dl.gen == s_eb_pool_gen &&
                        s_dl.cube == P->cube_mask && s_dl.ubo == s_eb.ub.buf &&
                        !memcmp(s_dl.v, rv, sizeof rv) && !memcmp(s_dl.s, rs, sizeof rs)) {
                        set = s_dl.set;
                        /* VK_CB_REUSE=1: no constants bound since the slices this
                         * set's previous draw uploaded, in this same ring generation:
                         * point at those again instead of copying the same bytes. */
                        if (vk_cb_reuse() && !s_e2.cb_dirty && s_dl.vsb == vsb && s_dl.psb == psb) {
                            s_eb.vb_head = vh; s_eb.ib_head = ih; s_eb.rb_head = rh;
                            dyn_off[0] = (uint32_t)s_dl.uvs; dyn_off[1] = (uint32_t)s_dl.ups;
                            s_dl_reused++; s_cb_reused++;
                            break;
                        }
                        s_eb.vb_head = vh; s_eb.ib_head = ih; s_eb.ub_head = uh; s_eb.rb_head = rh;
                        memcpy((u8*)s_eb.ub.ptr + uvs, vk_eng_ucpu(), vsb);
                        memcpy((u8*)s_eb.ub.ptr + ups, vk_eng_ucpu() + s_g.ps_off, psb);
                        dyn_off[0] = (uint32_t)uvs; dyn_off[1] = (uint32_t)ups;
                        s_dl.uvs = uvs; s_dl.ups = ups; s_dl.vsb = vsb; s_dl.psb = psb;
                        s_e2.cb_dirty = 0;
                        s_dl_reused++;
                        break;
                    }
                }
                VkDescriptorSetLayout lay = P->cube_mask ? s_e2.split_layout : s_g.set_layout;
                VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                    .descriptorPool = s_eb.pool, .descriptorSetCount = 1,
                                                    .pSetLayouts = &lay };
                const unsigned long long qa = dp[0] ? vk_dp_now() : 0;
                if (pvkAllocateDescriptorSets(s_vk.device, &dai, &set) == VK_SUCCESS) {
                    const unsigned long long qb = qa ? vk_dp_now() : 0;
                    s_eb.sets++;
                    s_eb.vb_head = vh; s_eb.ib_head = ih; s_eb.ub_head = uh; s_eb.rb_head = rh;
                    memcpy((u8*)s_eb.ub.ptr + uvs, vk_eng_ucpu(), vsb);
                    memcpy((u8*)s_eb.ub.ptr + ups, vk_eng_ucpu() + s_g.ps_off, psb);
                    if (qa) { s_dq_us[0] += qb - qa; s_dq_us[1] += vk_dp_now() - qb; }
                    if (!vk_eng_write_descriptors_to(P, set, s_eb.ub.buf, uvs, ups)) return -1;
                    dyn_off[0] = (uint32_t)uvs; dyn_off[1] = (uint32_t)ups;
                    if (vk_ub_dynamic()) {
                        s_dl.pool = s_eb.pool; s_dl.gen = s_eb_pool_gen; s_dl.cube = P->cube_mask;
                        s_dl.set = set; s_dl.ubo = s_eb.ub.buf;
                        memcpy(s_dl.v, rv, sizeof rv); memcpy(s_dl.s, rs, sizeof rs);
                        s_dl.uvs = uvs; s_dl.ups = ups; s_dl.vsb = vsb; s_dl.psb = psb;
                        s_e2.cb_dirty = 0;
                    }
                    break;
                }
                set = VK_NULL_HANDLE;
            }
            if (tries || !s_eb_pending) { batched = 0; break; }
            vk_eng_flush();
        }
        if (batched && !set) batched = 0;
    }
    if (job && !batched) return -1;                 /* the engine fetches on the CPU instead */
    dp[2] = dp[0] ? vk_dp_now() : 0;
    const int merge = batched && n <= 4u && vk_pass_merge();
    int keep = 0;                             /* record into the open pass */
    if (batched) {
        if (job) {
            u32* hw = (u32*)((u8*)s_eb.rb.ptr + roff);
            VkDeviceSize at = roff + (4u + 8u * job->nslots) * 4u;
            hw[0] = job->count; hw[1] = job->nslots;
            hw[2] = (u32)(voff / 4u); hw[3] = stride / 4u;
            VkDeviceSize gat[16];
            for (u32 g = 0; g < s_vd_ng; g++) {
                const size_t gb = (size_t)(s_vd_g1[g] - s_vd_g0[g]);
                gat[g] = at;
                memcpy((u8*)s_eb.rb.ptr + at, s_vd_g0[g], gb);
                at += ((VkDeviceSize)gb + 3u) & ~(VkDeviceSize)3u;
            }
            for (u32 s = 0; s < job->nslots; s++) {
                const rsx_be_vtx_slot* js = &job->slot[s];
                u32* d = hw + 4u + 8u * s;
                d[0] = js->type ? (u32)(gat[s_vd_grp[s]] + (VkDeviceSize)(js->src - s_vd_g0[s_vd_grp[s]])) : 0u;
                d[1] = js->stride; d[2] = js->type; d[3] = js->n;
                memcpy(&d[4], js->def, 16);
            }
            VkCommandBuffer pro = s_eb.slot[s_eb.cur].pro;
            if (!s_eb.slot[s_eb.cur].pro_open) {
                VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
                if (pvkResetCommandBuffer(pro, 0) != VK_SUCCESS || pvkBeginCommandBuffer(pro, &bi) != VK_SUCCESS)
                    return -1;
                pvkCmdBindPipeline(pro, VK_PIPELINE_BIND_POINT_COMPUTE, s_vd.pipe);
                pvkCmdBindDescriptorSets(pro, VK_PIPELINE_BIND_POINT_COMPUTE, s_vd.layout, 0, 1, &s_vd.set, 0, NULL);
                s_eb.slot[s_eb.cur].pro_open = 1;
            }
            const u32 hdr = (u32)(roff / 4u);
            pvkCmdPushConstants(pro, s_vd.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &hdr);
            pvkCmdDispatch(pro, (job->count * job->nslots + 63u) / 64u, 1, 1);
            s_vds.draws++;
        } else
        memcpy((u8*)s_eb.vb.ptr + voff, vertices, (size_t)vbytes);
        if (ibytes) memcpy((u8*)s_eb.ib.ptr + ioff, indices, (size_t)ibytes);
        dp[3] = dp[0] ? vk_dp_now() : 0;
        vbuf = s_eb.vb.buf; ibuf = s_eb.ib.buf;
        if (merge && s_pass.open && s_eb_pending && s_pass.rp == rp && s_pass.fb == fb &&
            s_pass.w == w && s_pass.h == h) {
            keep = 1;
            for (u32 u = 0; u < RSX_MAX_TEXTURES && keep; u++) {
                if (!((s_e2.tex_mask >> u) & 1u) || !s_e2.tex[u] || s_e2.tex[u] >= VK_ENG_MAX_OBJ) continue;
                const vk_eng_obj* o = &s_eobj[s_e2.tex[u]];
                const u32 src = o->kind == VK_ENG_VIEW ? o->surface : s_e2.tex[u];
                if (src == s_pass.depth) keep = 0;
                for (u32 i = 0; i < s_pass.n; i++)
                    if (src == s_pass.rt[i]) keep = 0;
            }
            if (!keep) s_pass_n[2]++;
        }
        if (!keep) {
            if (!s_eb_pending && vk_begin()) return -1;
            /* Earlier draws of this batch may have rendered what this one samples
             * or blends over: order it after all of them. */
            vk_cmd_after_everything(s_vk.cmd);
        }
    } else {
        if (s_eb_pending) vk_eng_flush();
        if (vk_ensure_vertex_buffer(vbytes)) return -1;
        memcpy(s_vk.vertices.ptr, vertices, (size_t)vbytes);
        if (ibytes) {
            if (vk_eng_ensure_indices(ibytes)) return -1;
            memcpy(s_e2.indices.ptr, indices, (size_t)ibytes);
        }
        vbuf = s_vk.vertices.buf; ibuf = s_e2.indices.buf; voff = ioff = 0;
        if (!vk_eng_ucpu()) return -1;
        memcpy(s_g.ubo.ptr, s_e2.ucpu, vsb);
        memcpy((u8*)s_g.ubo.ptr + s_g.ps_off, s_e2.ucpu + s_g.ps_off, psb);
        set = vk_eng_write_descriptors(P);
        if (!set) return -1;
        dyn_off[0] = 0; dyn_off[1] = (uint32_t)s_g.ps_off;
        if (vk_begin()) return -1;
        dp[3] = dp[0] ? vk_dp_now() : 0;
    }
    if (keep) {
        s_pass_n[1]++;
    } else {
        /* Everything submitted before is complete (each submission is waited
         * on); this orders memory for the surfaces this draw touches. */
        for (u32 i = 0; i < n; i++)
            vk_barrier_image(s_eobj[s_eng_bound.rt[i]].im.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
        VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                      .renderPass = rp, .framebuffer = fb,
                                      .renderArea = { { 0, 0 }, { w, h } } };
        pvkCmdBeginRenderPass(s_vk.cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        s_pass_n[0]++;
        if (merge) {
            s_pass.open = 1;
            s_pass.rp = rp; s_pass.fb = fb; s_pass.w = w; s_pass.h = h; s_pass.n = n;
            for (u32 i = 0; i < n; i++) s_pass.rt[i] = s_eng_bound.rt[i];
            s_pass.depth = dkey;
        }
    }
    /* VK_STATE_SKIP=1: state this command buffer already holds from the
     * previous draw (same epoch: no other path set any since) is not set again.
     * Exact: dynamic state and the bound pipeline persist across render passes
     * within one command buffer. */
    static int s_skip = -1;
    if (s_skip < 0) { const char* e = getenv("VK_STATE_SKIP"); s_skip = e ? atoi(e) : 0; }
    static struct { VkCommandBuffer cmd; u32 epoch; VkPipeline pipe; VkViewport vp; VkRect2D sc;
                    u32 sref; float db[2]; float dbias[2]; float bc[4]; } s_rs;
    const int same = s_skip && s_rs.cmd == s_vk.cmd && s_rs.epoch == s_rec_epoch;
    if (!same) { memset(&s_rs, 0, sizeof s_rs); s_rs.cmd = s_vk.cmd; s_rs.epoch = s_rec_epoch; s_rs.sref = ~0u; }
    if (!same || s_rs.pipe != pipe) { pvkCmdBindPipeline(s_vk.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); s_rs.pipe = pipe; }
    VkViewport vp = { 0.0f, 0.0f, (float)w, (float)h, 0.0f, 1.0f };
    if (s_e2.have_vp && s_e2.vp[2] > 0.0f && s_e2.vp[3] > 0.0f) {
        vp.x = s_e2.vp[0]; vp.y = s_e2.vp[1]; vp.width = s_e2.vp[2]; vp.height = s_e2.vp[3];
    }
    VkRect2D sc = { { 0, 0 }, { w, h } };
    if (s_e2.have_sc) {                       /* clamped: Vulkan rejects negative or over-size */
        u32 x0 = s_e2.sc[0] < w ? s_e2.sc[0] : w, y0 = s_e2.sc[1] < h ? s_e2.sc[1] : h;
        u32 x1 = s_e2.sc[0] + s_e2.sc[2], y1 = s_e2.sc[1] + s_e2.sc[3];
        if (x1 > w) x1 = w;
        if (y1 > h) y1 = h;
        sc.offset.x = (int32_t)x0; sc.offset.y = (int32_t)y0;
        sc.extent.width = x1 > x0 ? x1 - x0 : 0; sc.extent.height = y1 > y0 ? y1 - y0 : 0;
    }
    if (!same || memcmp(&s_rs.vp, &vp, sizeof vp)) { pvkCmdSetViewport(s_vk.cmd, 0, 1, &vp); s_rs.vp = vp; }
    if (!same || memcmp(&s_rs.sc, &sc, sizeof sc)) { pvkCmdSetScissor(s_vk.cmd, 0, 1, &sc); s_rs.sc = sc; }
    if (!same || s_rs.sref != (s_e2.stencil_ref & 0xFFu)) {
        pvkCmdSetStencilReference(s_vk.cmd, VK_STENCIL_FACE_FRONT_AND_BACK, s_e2.stencil_ref & 0xFFu);
        s_rs.sref = s_e2.stencil_ref & 0xFFu;
    }
    if (!same || memcmp(s_rs.bc, s_e2.blend_color, sizeof s_rs.bc)) {
        pvkCmdSetBlendConstants(s_vk.cmd, s_e2.blend_color);
        memcpy(s_rs.bc, s_e2.blend_color, sizeof s_rs.bc);
    }
    if (!same || memcmp(s_rs.dbias, s_e2.depth_bias, sizeof s_rs.dbias)) {
        pvkCmdSetDepthBias(s_vk.cmd, s_e2.depth_bias[1], 0.0f, s_e2.depth_bias[0]);
        memcpy(s_rs.dbias, s_e2.depth_bias, sizeof s_rs.dbias);
    }
    if (s_vk.depth_bounds_ok && (!same || memcmp(s_rs.db, s_e2.depth_bounds, sizeof s_rs.db))) {
        pvkCmdSetDepthBounds(s_vk.cmd, s_e2.depth_bounds[0], s_e2.depth_bounds[1]);
        memcpy(s_rs.db, s_e2.depth_bounds, sizeof s_rs.db);
    }
    pvkCmdBindDescriptorSets(s_vk.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                             P->cube_mask ? s_e2.split_pipe_layout : s_g.pipe_layout,
                             0, 1, &set, vk_ub_dynamic() ? 2u : 0u, dyn_off);
    pvkCmdBindVertexBuffers(s_vk.cmd, 0, 1, &vbuf, &voff);
    if (ibytes) {
        pvkCmdBindIndexBuffer(s_vk.cmd, ibuf, ioff, VK_INDEX_TYPE_UINT32);
        pvkCmdDrawIndexed(s_vk.cmd, index_count, 1, 0, 0, 0);
    } else {
        pvkCmdDraw(s_vk.cmd, vertex_count, 1, 0, 0);
    }
    if (!merge) pvkCmdEndRenderPass(s_vk.cmd);
    dp[4] = dp[0] ? vk_dp_now() : 0;
    if (!batched) {
        vk_submit_and_wait();
    } else {
        s_eb.draws++;
        if (++s_eb_pending >= s_eb.max_draws) vk_eng_flush();
    }
    if (dp[0]) { dp[5] = vk_dp_now(); vk_dp_end(dp); }
    if (job && check) {
        /* The vertex ring is host-coherent: once every batch is done the
         * decoded slice can be compared in place with the CPU's vertices. */
        vk_eng_flush();
        vk_eng_idle();
        s_vds.checked++;
        if (memcmp((u8*)s_eb.vb.ptr + voff, check, (size_t)vbytes)) {
            if (s_vds.differ++ < 4) {
                const u8* g = (const u8*)s_eb.vb.ptr + voff; const u8* c = (const u8*)check;
                size_t k = 0; while (k < vbytes && g[k] == c[k]) k++;
                u32 gw = 0, cw = 0; memcpy(&gw, g + (k & ~(size_t)3), 4); memcpy(&cw, c + (k & ~(size_t)3), 4);
                VK_LOG("vtx-gpu check: %u vertices x %u slots differ at byte %zu: gpu 0x%08X cpu 0x%08X\n",
                       job->count, job->nslots, k, gw, cw);
            }
        }
    }
    if (job) {
        const time_t now = time(NULL);
        if (!s_vds.t) s_vds.t = now;
        if (now - s_vds.t >= 60) {
            VK_LOG("vtx-gpu 60s: %llu draws decoded on the GPU; checked %llu, %llu differ\n",
                   s_vds.draws, s_vds.checked, s_vds.differ);
            memset(&s_vds, 0, sizeof s_vds); s_vds.t = now;
        }
    }
    return 0;
}

static void vk_eng_draw(void* user, rsx_topology topology, const void* vertices, u32 vertex_count,
                        u32 stride, const u32* indices, u32 index_count)
{
    (void)user;
    vk_eng_draw_impl(topology, vertices, vertex_count, stride, indices, index_count, NULL, NULL);
}

static int vk_eng_draw_gpu(void* user, rsx_topology topology, const rsx_be_vtx_job* job, u32 stride,
                           const u32* indices, u32 index_count, const void* check)
{
    (void)user;
    return vk_eng_draw_impl(topology, NULL, 0, stride, indices, index_count, job, check);
}

static const rsx_draw_backend s_vk_engine_backend = {
    .user                 = NULL,
    .init                 = vk_eng_init,
    .shutdown             = vk_eng_shutdown,
    .submit_and_wait      = vk_eng_submit_and_wait,
    .texture_create       = vk_eng_texture_create,
    .texture_upload       = vk_eng_texture_upload,
    .texture_release      = vk_eng_release,
    .color_target_create  = vk_eng_color_target_create,
    .color_target_release = vk_eng_release,
    .surface_view         = vk_eng_surface_view,
    .depth_target_create  = vk_eng_depth_target_create,
    .depth_target_release = vk_eng_release,
    .depth_snapshot       = vk_eng_depth_snapshot,
    .color_snapshot       = vk_eng_color_snapshot,
    .pipeline_create      = vk_eng_pipeline_create,
    .pipeline_release     = vk_eng_pipeline_release,
    .bind_targets         = vk_eng_bind_targets,
    .bind_pipeline        = vk_eng_bind_pipeline,
    .bind_vs_constants    = vk_eng_bind_vs_constants,
    .bind_ps_constants    = vk_eng_bind_ps_constants,
    .bind_textures        = vk_eng_bind_textures,
    .bind_vertex_textures = vk_eng_bind_vertex_textures,
    .set_viewport         = vk_eng_set_viewport,
    .set_scissor          = vk_eng_set_scissor,
    .set_stencil_ref      = vk_eng_set_stencil_ref,
    .set_blend_color      = vk_eng_set_blend_color,
    .pipeline_prewarm     = vk_eng_pipeline_prewarm,
    .pipeline_ready       = vk_eng_pipeline_ready,
    .pipeline_variants    = vk_eng_pipeline_variants,
    .set_depth_bounds     = vk_eng_set_depth_bounds,
    .set_depth_bias       = vk_eng_set_depth_bias,
    .texture_create_3d    = vk_eng_texture_create_3d,
    .texture_upload_3d    = vk_eng_texture_upload_3d,
    .depth_bounds_native  = vk_eng_depth_bounds_native,
    .draw                 = vk_eng_draw,
    .clear_color          = vk_eng_clear_color,
    .clear_depth_stencil  = vk_eng_clear_depth_stencil,
    .present              = vk_eng_present,
    .readback             = vk_eng_readback,
    .depth_readback       = vk_eng_depth_readback,
    .stencil_readback     = vk_eng_stencil_readback,
    .depth_to_argb8       = vk_eng_depth_to_argb8,
    .color_to_rg16f       = vk_eng_color_to_rg16f,
    .image_copy           = vk_eng_image_copy,
    .draw_gpu_vertices    = vk_eng_draw_gpu,
    .color_target_write   = vk_eng_color_target_write,
    /* Batched uploads record behind vk_cmd_after_everything; unbatched ones
     * submit after the open batch is flushed. */
    .texture_upload_ordered = 1,
};

static rsx_backend s_vulkan_backend = {
    .userdata           = &s_vk,
    .set_render_target  = vk_cb_track_state,
    .set_vertex_attribs = vk_cb_track_state,
    .set_depth_stencil  = vk_cb_track_state,
    .clear              = vk_cb_clear,
    .draw_arrays        = vk_cb_draw_arrays,
    .draw_indexed       = vk_cb_draw_indexed,
    .bind_texture       = vk_cb_bind_texture,
    .present            = vk_cb_present,
};

/* ---------------------------------------------------------------------------
 * Public entry points
 * -------------------------------------------------------------------------*/
static int vk_init_all(u32 width, u32 height, const char* title)
{
    pthread_mutex_init(&s_vk.queue_mu, NULL);
    pthread_mutex_init(&s_vk.present_mu, NULL);
    s_vk.width  = width  ? width  : 1280;
    s_vk.height = height ? height : 720;
    s_vk.dump_path = getenv("PS3RECOMP_VK_DUMP");
    /* Writing a full frame to disk on every present costs more than the
     * present itself and slows a title enough to change what it does; a
     * live title wants one now and then, not all of them. */
    { const char* e = getenv("PS3RECOMP_VK_DUMP_EVERY");
      s_vk.dump_every = (e && atoi(e) > 0) ? (u32)atoi(e) : 1; }

    vk_window_open(title);           /* no-op unless PS3RECOMP_VK_WINDOW is set */
    if (vk_load_library()) return -1;

    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                              .pApplicationName = "ps3recomp", .pEngineName = "ps3recomp RSX",
                              .apiVersion = VK_API_VERSION_1_0 };
    VkInstanceCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
    if (s_vk.windowed) {
        ci.enabledExtensionCount   = s_vk.inst_ext_count;
        ci.ppEnabledExtensionNames = s_vk.inst_exts;
    }
#ifdef __APPLE__
    /* MoltenVK is a portability driver: the loader enumerates it only for an
     * instance that asks for portability devices. */
    const char* port_exts[17];
    u32 n_port = 0;
    for (u32 i = 0; i < ci.enabledExtensionCount && n_port < 16; i++) port_exts[n_port++] = ci.ppEnabledExtensionNames[i];
    port_exts[n_port++] = "VK_KHR_portability_enumeration";
    ci.enabledExtensionCount   = n_port;
    ci.ppEnabledExtensionNames = port_exts;
    ci.flags |= 0x00000001;   /* VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR */
#endif
    VkResult ir = pvkCreateInstance(&ci, NULL, &s_vk.instance);
    if (ir != VK_SUCCESS && s_vk.windowed) {
        vk_window_drop("vkCreateInstance with the surface extensions failed");
#ifdef __APPLE__
        ci.enabledExtensionCount = 1;
        ci.ppEnabledExtensionNames = &port_exts[n_port - 1];
#else
        ci.enabledExtensionCount = 0;
        ci.ppEnabledExtensionNames = NULL;
#endif
        ir = pvkCreateInstance(&ci, NULL, &s_vk.instance);
    }
    if (ir != VK_SUCCESS) { VK_LOG("vkCreateInstance failed (VkResult %d)\n", (int)ir); return -1; }
    if (vk_load_instance_functions()) return -1;
    if (s_vk.windowed) {
        if (vk_load_wsi_instance_functions()) {
            vk_window_drop("the driver has no surface functions");
        } else if (!SDL_Vulkan_CreateSurface(s_vk.window, s_vk.instance, &s_vk.surface)) {
            char why[256];
            snprintf(why, sizeof why, "SDL_Vulkan_CreateSurface failed: %s", SDL_GetError());
            vk_window_drop(why);
        }
    }
    if (vk_pick_device()) {
        if (!s_vk.windowed) return -1;
        vk_window_drop("no device can present to the window");
        if (vk_pick_device()) return -1;
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                    .queueFamilyIndex = s_vk.queue_family,
                                    .queueCount = 1, .pQueuePriorities = &prio };
    const char* dev_exts[4];
    u32 n_dev_exts = 0;
    if (s_vk.windowed) dev_exts[n_dev_exts++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
#ifdef __APPLE__
    /* A portability device must have VK_KHR_portability_subset enabled. */
    dev_exts[n_dev_exts++] = "VK_KHR_portability_subset";
#endif
    /* RSX's MIRROR_ONCE wrap modes (see vk_gcm_wrap). */
    { const char* e = getenv("RSX_VK_MIRROR_ONCE");
      if ((!e || atoi(e) != 0) && vk_has_device_extension(s_vk.phys, "VK_KHR_sampler_mirror_clamp_to_edge")) {
          dev_exts[n_dev_exts++] = "VK_KHR_sampler_mirror_clamp_to_edge";
          s_vk.mirror_clamp_ok = 1;
      }
      fprintf(stderr, "[vulkan] MIRROR_ONCE wrap: %s\n", s_vk.mirror_clamp_ok ? "mirror clamp to edge"
                                                                                 : "clamp to edge (approximation)"); }
    /* DXT (BC1-3) is how PS3 titles ship most textures; Vulkan gates it
     * behind a feature, enabled only when the device reports it. */
    VkPhysicalDeviceFeatures have, want;
    memset(&want, 0, sizeof want);
    pvkGetPhysicalDeviceFeatures(s_vk.phys, &have);
    want.textureCompressionBC = have.textureCompressionBC;
    s_vk.bc_ok = have.textureCompressionBC ? 1 : 0;
    /* The depth bounds test: a deferred renderer bounds its light pass to
     * the geometry and its sky to the far plane with it. Optional in
     * Vulkan; without it the test stays off, as before. */
    want.depthBounds = have.depthBounds;
    s_vk.depth_bounds_ok = have.depthBounds ? 1 : 0;
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                               .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
                               .enabledExtensionCount = n_dev_exts,
                               .ppEnabledExtensionNames = n_dev_exts ? dev_exts : NULL,
                               .pEnabledFeatures = &want };
    VK_CHECK(pvkCreateDevice(s_vk.phys, &dci, NULL, &s_vk.device), "vkCreateDevice");
    if (vk_load_device_functions()) return -1;
    if (s_vk.windowed && vk_load_wsi_device_functions())
        vk_window_drop("the driver has no swapchain functions");
    pvkGetDeviceQueue(s_vk.device, s_vk.queue_family, 0, &s_vk.queue);

    VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                    .queueFamilyIndex = s_vk.queue_family };
    VK_CHECK(pvkCreateCommandPool(s_vk.device, &pci, NULL, &s_vk.pool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                        .commandPool = s_vk.pool,
                                        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                        .commandBufferCount = 1 };
    VK_CHECK(pvkAllocateCommandBuffers(s_vk.device, &cai, &s_vk.cmd), "vkAllocateCommandBuffers");
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VK_CHECK(pvkCreateFence(s_vk.device, &fci, NULL, &s_vk.fence), "vkCreateFence");
    /* The present's own (see vk_state). */
    VK_CHECK(pvkCreateCommandPool(s_vk.device, &pci, NULL, &s_vk.present_pool), "vkCreateCommandPool");
    cai.commandPool = s_vk.present_pool;
    VK_CHECK(pvkAllocateCommandBuffers(s_vk.device, &cai, &s_vk.present_cmd), "vkAllocateCommandBuffers");
    VK_CHECK(pvkCreateFence(s_vk.device, &fci, NULL, &s_vk.present_fence), "vkCreateFence");
    VK_CHECK(pvkCreateFence(s_vk.device, &fci, NULL, &s_vk.window_fence), "vkCreateFence");

    if (vk_pick_depth_format()) return -1;
    if (vk_create_image(&s_vk.color, VK_FORMAT_R8G8B8A8_UNORM, s_vk.width, s_vk.height,
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT)) return -1;
    if (vk_create_image(&s_vk.depth, s_vk.depth_format, s_vk.width, s_vk.height,
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, s_vk.depth_aspect)) return -1;
    if (vk_create_host_buffer(&s_vk.readback, (VkDeviceSize)s_vk.width * s_vk.height * 4u,
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return -1;
    if (vk_create_render_pass()) return -1;
    if (vk_create_pipeline_objects()) return -1;
    if (vk_guest_init()) {
        VK_LOG("guest programs: setup failed, fixed-function path only\n");
        vk_guest_shutdown();
    }

    /* Both targets UNDEFINED -> GENERAL once. Colour starts opaque black so a
     * present before the first clear reads defined memory; depth starts at
     * the FAR plane, not zero -- zero is the near plane, and a guest that
     * enables LESS before its first depth clear would draw nothing (the
     * null backend makes the same choice). */
    if (vk_begin()) return -1;
    vk_barrier_image(s_vk.color.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
    vk_barrier_image(s_vk.depth.img, s_vk.depth_aspect, VK_IMAGE_LAYOUT_UNDEFINED);
    VkClearColorValue black = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } };
    VkClearDepthStencilValue far_plane = { 1.0f, 0 };
    VkImageSubresourceRange cr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VkImageSubresourceRange dr = { s_vk.depth_aspect, 0, 1, 0, 1 };
    pvkCmdClearColorImage(s_vk.cmd, s_vk.color.img, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &cr);
    pvkCmdClearDepthStencilImage(s_vk.cmd, s_vk.depth.img, VK_IMAGE_LAYOUT_GENERAL, &far_plane, 1, &dr);
    if (vk_submit_and_wait()) return -1;

    if (s_vk.windowed) {
        VkSemaphoreCreateInfo semi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        if (pvkCreateSemaphore(s_vk.device, &semi, NULL, &s_vk.sem_acquire) != VK_SUCCESS ||
            vk_create_swapchain())
            vk_window_drop("the swapchain could not be created");
    }
    return 0;
}

int rsx_vulkan_backend_init(u32 width, u32 height, const char* title)
{
    memset(&s_vk, 0, sizeof s_vk);
    if (vk_init_all(width, height, title)) {
        rsx_vulkan_backend_shutdown();
        return -1;
    }
    /* One path at a time: the FIFO walker feeds both the rsx_state vtable
     * and the register-file engine, so registering both would record every
     * draw twice (the rule the Metal backend's init follows).
     *
     * With guest programs on, the engine is the default -- as it became
     * Metal's once every host mode passed through it. It builds every
     * pipeline from the guest's programs, so without them it has nothing to
     * draw with, and the vtable's fixed-function path stays the default.
     * Either way PS3RECOMP_RSX_ENGINE=dispatch|vtable overrides. */
    rsx_draw_engine_set_backend(&s_vk_engine_backend);
    rsx_draw_engine_set_default(s_g.on ? 1 : 0);
    if (rsx_draw_engine_enabled() && rsx_draw_engine_init(s_vk.width, s_vk.height) == 0)
        s_vk.eng_active = 1;
    else
        rsx_set_backend(&s_vulkan_backend);
    VK_LOG("%s path\n", s_vk.eng_active ? "register-file draw engine" : "vtable");
    VK_LOG("offscreen target %ux%u ready, depth format %d (%s)\n",
           s_vk.width, s_vk.height, (int)s_vk.depth_format, title ? title : "");
    return 0;
}

void rsx_vulkan_backend_shutdown(void)
{
    rsx_vulkan_backend_present_finish();   /* a host that began one and quit */
    if (rsx_get_backend() == &s_vulkan_backend) rsx_set_backend(NULL);
    if (s_vk.eng_active) { rsx_draw_engine_shutdown(); s_vk.eng_active = 0; }
    rsx_draw_engine_set_backend(NULL);
    if (s_vk.windowed && s_vk.swapchain && s_vk.presented && s_vk.hold_seconds > 0.0) {
        VK_LOG("holding the last frame on screen for %.1f s "
               "(Esc or closing the window ends it early)\n", s_vk.hold_seconds);
        const Uint32 end = SDL_GetTicks() + (Uint32)(s_vk.hold_seconds * 1000.0);
        while (s_vk.windowed && !SDL_TICKS_PASSED(SDL_GetTicks(), end)) {
            if (rsx_vulkan_backend_pump_messages() != 0) break;
            vk_present_window();         /* FIFO: paced by the display */
        }
    }
    if (s_vk.windowed || s_vk.window) vk_window_drop(NULL);
    if (s_vk.device) {
        vk_device_idle();
        vk_guest_shutdown();
        for (int i = 0; i < VK_PIPELINE_SLOTS; i++)
            if (s_vk.pipelines[i]) pvkDestroyPipeline(s_vk.device, s_vk.pipelines[i], NULL);
        if (s_vk.pipe_layout) pvkDestroyPipelineLayout(s_vk.device, s_vk.pipe_layout, NULL);
        if (s_vk.desc_pool)   pvkDestroyDescriptorPool(s_vk.device, s_vk.desc_pool, NULL);
        if (s_vk.set_layout)  pvkDestroyDescriptorSetLayout(s_vk.device, s_vk.set_layout, NULL);
        if (s_vk.sampler)     pvkDestroySampler(s_vk.device, s_vk.sampler, NULL);
        for (u32 i = 0; i < s_vk.samp_count; i++)
            pvkDestroySampler(s_vk.device, s_vk.samp_cache[i].s, NULL);
        if (s_vk.vs)          pvkDestroyShaderModule(s_vk.device, s_vk.vs, NULL);
        if (s_vk.fs)          pvkDestroyShaderModule(s_vk.device, s_vk.fs, NULL);
        if (s_vk.framebuffer) pvkDestroyFramebuffer(s_vk.device, s_vk.framebuffer, NULL);
        if (s_vk.render_pass) pvkDestroyRenderPass(s_vk.device, s_vk.render_pass, NULL);
        for (u32 u = 0; u < RSX_MAX_TEXTURES; u++) vk_destroy_image(&s_vk.tex[u]);
        vk_destroy_image(&s_vk.dummy);
        vk_destroy_image(&s_vk.depth);
        vk_destroy_image(&s_vk.color);
        vk_destroy_buffer(&s_vk.vertices);
        vk_destroy_buffer(&s_vk.readback);
        if (s_eb.on == 1) {            /* the rings, then the batch slots (slot 0 is s_vk's own) */
            vk_destroy_buffer(&s_eb.vb);
            vk_destroy_buffer(&s_eb.ib);
            vk_destroy_buffer(&s_eb.ub);
            vk_destroy_buffer(&s_eb.sb);
            s_vk.cmd = s_eb.slot[0].cmd;
            s_vk.fence = s_eb.slot[0].fence;
            for (u32 i = 0; i < VK_EB_SLOTS; i++) {
                if (s_eb.slot[i].pool) pvkDestroyDescriptorPool(s_vk.device, s_eb.slot[i].pool, NULL);
                if (i && s_eb.slot[i].fence) pvkDestroyFence(s_vk.device, s_eb.slot[i].fence, NULL);
            }
        }
        if (s_vk.fence) pvkDestroyFence(s_vk.device, s_vk.fence, NULL);
        if (s_vk.pool)  pvkDestroyCommandPool(s_vk.device, s_vk.pool, NULL);
        if (s_vk.present_fence) pvkDestroyFence(s_vk.device, s_vk.present_fence, NULL);
        if (s_vk.window_fence)  pvkDestroyFence(s_vk.device, s_vk.window_fence, NULL);
        if (s_vk.present_pool)  pvkDestroyCommandPool(s_vk.device, s_vk.present_pool, NULL);
        pvkDestroyDevice(s_vk.device, NULL);
    }
    if (s_vk.instance && pvkDestroyInstance) pvkDestroyInstance(s_vk.instance, NULL);
    if (s_vk.lib) VK_DLCLOSE(s_vk.lib);
    free(s_vk.cpu_verts);
    pthread_mutex_destroy(&s_vk.queue_mu);
    pthread_mutex_destroy(&s_vk.present_mu);
    memset(&s_vk, 0, sizeof s_vk);
}

int rsx_vulkan_backend_pump_messages(void)
{
    if (!s_vk.windowed) return 0;
    int quit = 0;
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        /* Closing the window (or Cmd+Q) quits. Esc does not: a player reaches for
         * it to pause or skip, and it took the whole session with it; the pad's
         * keyboard fallback reads it as START. */
        if (e.type == SDL_QUIT) quit = 1;
    }
    return quit ? -1 : 0;
}

void rsx_vulkan_backend_present(void)
{
    rsx_vulkan_backend_present_begin();
    rsx_vulkan_backend_present_finish();
}

void rsx_vulkan_backend_present_begin(void)
{
    if (!s_vk.device) return;
    s_vk.present_split = 1;
    if (s_vk.eng_active) rsx_draw_engine_present();
    else                 vk_cb_present(&s_vk, 0);
    s_vk.present_split = 0;
}

void rsx_vulkan_backend_present_finish(void)
{
    if (!s_vk.device) return;
    pthread_mutex_lock(&s_vk.present_mu);
    vk_present_complete_locked();
    pthread_mutex_unlock(&s_vk.present_mu);
}

u32 rsx_vulkan_backend_presented_center(void)
{
    return s_vk.presented ? (0xFF000000u | s_vk.last_center) : 0u;
}

u32 rsx_vulkan_backend_debug_color(void) { return s_vk.clear_argb; }

u32 rsx_vulkan_backend_readback_center(void)
{
    if (s_vk.eng_active) return rsx_draw_engine_readback_center();
    return s_vk.presented ? (0xFF000000u | s_vk.last_center) : 0u;
}

u32 rsx_vulkan_backend_guest_draws(void)
{
    return s_vk.eng_active ? rsx_draw_engine_guest_draws() : s_g.last_draws;
}

int rsx_vulkan_backend_depth_bounds(void)
{
    return s_vk.eng_active && s_vk.depth_bounds_ok;
}
