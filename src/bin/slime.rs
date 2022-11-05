use core::time;
use std::f32::consts::PI;
use std::io::{Cursor, Write};
use std::iter::repeat;
use std::mem::size_of;
use std::sync::Arc;
use std::thread::current;
use std::time::{SystemTime, UNIX_EPOCH};

use log::{error, info};

use anyhow::Result;

use bytemuck::{Pod, Zeroable};
use rand::Rng;
use vulkano::buffer::{BufferContents, BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageInfo, PrimaryAutoCommandBuffer,
    RenderPassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{
    ImageDimensions, ImageUsage, ImmutableImage, MipmapsCount, StorageImage, SwapchainImage,
};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};

use vulkano::shader::ShaderModule;
use vulkano::swapchain::{
    self, AcquireError, PresentInfo, PresentMode, Surface, Swapchain, SwapchainCreateInfo,
    SwapchainCreationError,
};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::VulkanLibrary;

use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use vulkan::settings::Settings;

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct Vertex {
    position: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Zeroable, Pod)]
struct Agent {
    position: [f32; 2],
    direction: [f32; 2],
    // velocity: f32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Zeroable, Pod)]
struct TimeData {
    time: u32,
    delta_time: f32,
}

static WIDTH: u32 = 1920;
static HEIGHT: u32 = 1080;
static AGENTS: u32 = 1024 * 500;
static STEP_PER_FRAME: u32 = 1;
static RADIUS: f32 = HEIGHT as f32 / 20.0;

vulkano::impl_vertex!(Vertex, position);

mod bs {
    vulkano_shaders::shader!(ty: "compute", path: "shaders/blur.glsl");
}

mod cs {
    vulkano_shaders::shader!(ty: "compute", path: "shaders/agent.glsl");
}

mod vs {
    vulkano_shaders::shader!(ty: "vertex", path: "shaders/vertex.glsl");
}

mod fs {
    vulkano_shaders::shader!(ty: "fragment", path: "shaders/simple_texture.glsl");
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    let event_loop = EventLoop::new();
    let (device, queue, surface) = init_vulkan_and_surface(&event_loop)?;

    let caps = device
        .physical_device()
        .surface_capabilities(&surface, Default::default())
        .expect("failed to get surface capabilities");

    let dimensions = surface.window().inner_size();
    let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
    let image_format = Some(
        device
            .physical_device()
            .surface_formats(&surface, Default::default())?[0]
            .0,
    );

    let (mut swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1,
            image_format,
            image_extent: dimensions.into(),
            image_usage: ImageUsage {
                color_attachment: true,
                ..Default::default()
            },
            composite_alpha,
            present_mode: PresentMode::Immediate,
            ..Default::default()
        },
    )?;

    let settings_buf = CpuAccessibleBuffer::from_data(
        device.clone(),
        BufferUsage {
            uniform_buffer: true,
            ..Default::default()
        },
        false,
        Settings::get()?,
    )?;

    let disp_image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: WIDTH,
            height: HEIGHT,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(queue.queue_family_index()),
    )?;

    let compute_image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: WIDTH,
            height: HEIGHT,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(queue.queue_family_index()),
    )?;

    let blur_image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: WIDTH,
            height: HEIGHT,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(queue.queue_family_index()),
    )?;

    let mut last_frame = SystemTime::now();

    let time_buf = CpuAccessibleBuffer::from_data(
        device.clone(),
        BufferUsage {
            uniform_buffer: true,
            ..Default::default()
        },
        false,
        TimeData {
            time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u32,
            delta_time: 0.0,
        },
    )?;

    let mut rng = rand::thread_rng();

    let agents: Vec<Agent> = (0..AGENTS)
        .map(|i| {
            let angle = i as f32 / AGENTS as f32 * 2.0 * PI * 100.0;
            let radius = i as f32 / AGENTS as f32 * RADIUS;
            let pos_x = (WIDTH as f32 / 2.0) + angle.cos() * radius;
            let pos_y = (HEIGHT as f32 / 2.0) + angle.sin() * radius;

            // let pos_x = rng.gen::<f32>() * RADIUS + WIDTH as f32 / 2.0;
            // let pos_y = rng.gen::<f32>() * RADIUS + HEIGHT as f32 / 2.0;

            let dir_x = WIDTH as f32 / 2.0 - pos_x;
            let dir_y = HEIGHT as f32 / 2.0 - pos_y;

            let norm = dir_x * dir_x + dir_y * dir_y;

            Agent {
                position: [pos_x, pos_y],
                direction: [dir_x / norm, dir_y / norm],
            }
        })
        .collect();

    let agent_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage {
            storage_buffer: true,
            ..Default::default()
        },
        false,
        agents,
    )?;

    let cs = cs::load(device.clone())?;
    let agent_pipeline = ComputePipeline::new(
        device.clone(),
        cs.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )?;

    let dst_image_view = ImageView::new_default(compute_image.clone())?;
    let blur_image_view = ImageView::new_default(blur_image.clone())?;

    let layout = agent_pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, agent_buffer.clone()),
            WriteDescriptorSet::image_view(1, dst_image_view.clone()),
            WriteDescriptorSet::image_view(2, blur_image_view.clone()),
            WriteDescriptorSet::buffer(3, time_buf.clone()),
            WriteDescriptorSet::buffer(4, settings_buf.clone()),
        ],
    )?;

    let bs = bs::load(device.clone())?;
    let blur_pipeline = ComputePipeline::new(
        device.clone(),
        bs.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )?;

    let layout = blur_pipeline.layout().set_layouts().get(0).unwrap();
    let blur_set = PersistentDescriptorSet::new(
        layout.clone(),
        [
            WriteDescriptorSet::image_view(0, dst_image_view.clone()),
            WriteDescriptorSet::image_view(1, blur_image_view.clone()),
            WriteDescriptorSet::buffer(2, settings_buf.clone()),
            WriteDescriptorSet::buffer(3, time_buf.clone()),
        ],
    )?;

    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::MultipleSubmit,
    )?;

    for _ in 0..STEP_PER_FRAME {
        builder
            .bind_pipeline_compute(agent_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                agent_pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .dispatch([AGENTS / 64, 1, 1])?
            .bind_pipeline_compute(blur_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                blur_pipeline.layout().clone(),
                0,
                blur_set.clone(),
            )
            .dispatch([WIDTH / 8, HEIGHT / 8, 1])?
            .copy_image(CopyImageInfo::images(
                blur_image.clone(),
                compute_image.clone(),
            ))?;
    }

    builder.copy_image(CopyImageInfo::images(
        blur_image.clone(),
        disp_image.clone(),
    ))?;

    let compute_command_buffer = Arc::new(builder.build()?);

    let vertex1 = Vertex {
        position: [-1.0, -1.0],
    };

    let vertex2 = Vertex {
        position: [1.0, -1.0],
    };

    let vertex3 = Vertex {
        position: [1.0, 1.0],
    };

    let vertex4 = Vertex {
        position: [-1.0, -1.0],
    };

    let vertex5 = Vertex {
        position: [1.0, 1.0],
    };

    let vertex6 = Vertex {
        position: [-1.0, 1.0],
    };

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [1024.0, 1024.0],
        depth_range: 0.0..1.0,
    };

    let vs = vs::load(device.clone())?;
    let fs = fs::load(device.clone())?;

    let vertex_buf = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage {
            vertex_buffer: true,
            ..Default::default()
        },
        false,
        vec![vertex1, vertex2, vertex3, vertex4, vertex5, vertex6].into_iter(),
    )?;

    let render_pass = vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )?;

    let pipeline = get_pipeline(
        device.clone(),
        vs.clone(),
        fs.clone(),
        render_pass.clone(),
        viewport.clone(),
    )?;

    let frambuffers = get_framebuffers(&images, &render_pass);

    let disp_image_view = ImageView::new_default(disp_image.clone())?;
    let mut command_buffers = get_render_command_buffers(
        &device,
        &queue,
        &pipeline,
        &frambuffers,
        &vertex_buf,
        &disp_image_view,
    );

    let mut window_resized = false;
    let mut recrate_swapchain = false;

    // let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
            info!("Clean exit");
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => window_resized = true,
        Event::MainEventsCleared => {
            if window_resized || recrate_swapchain {
                recrate_swapchain = false;

                let new_dimensions = surface.window().inner_size();

                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: new_dimensions.into(), // here, "image_extend" will correspond to the window dimensions
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    // This error tends to happen when the user is manually resizing the window.
                    // Simply restarting the loop is the easiest way to fix this issue.
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                };

                swapchain = new_swapchain;
                let new_framebuffers = get_framebuffers(&new_images, &render_pass);

                if window_resized {
                    window_resized = false;

                    viewport.dimensions = new_dimensions.into();
                    let new_pipeline = get_pipeline(
                        device.clone(),
                        vs.clone(),
                        fs.clone(),
                        render_pass.clone(),
                        viewport.clone(),
                    )
                    .unwrap();

                    command_buffers = get_render_command_buffers(
                        &device,
                        &queue,
                        &new_pipeline,
                        &new_framebuffers,
                        &vertex_buf,
                        &disp_image_view,
                    );
                }
            }

            let (image_i, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recrate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image : {:?}", e),
                };

            if suboptimal {
                recrate_swapchain = true;
            }

            {
                let mut t = time_buf.write().unwrap();
                t.time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u32;

                let current_frame = SystemTime::now();
                let elapsed = current_frame
                    .duration_since(last_frame)
                    .unwrap()
                    .as_secs_f32();
                last_frame = current_frame;
                t.delta_time = elapsed;
            }

            let execution = sync::now(device.clone())
                .join(acquire_future)
                .then_execute(queue.clone(), compute_command_buffer.clone())
                .unwrap()
                .then_execute(queue.clone(), command_buffers[image_i].clone())
                .unwrap()
                .then_swapchain_present(
                    queue.clone(),
                    PresentInfo {
                        index: image_i,
                        ..PresentInfo::swapchain(swapchain.clone())
                    },
                )
                .then_signal_fence_and_flush();

            match execution {
                Ok(future) => {
                    future.wait(None).unwrap();
                    // previous_frame_end = Some(future.boxed());
                }
                Err(FlushError::OutOfDate) => {
                    recrate_swapchain = true;
                    // previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
                Err(e) => {
                    error!("Failed to flush future : {:?}", e);
                    // previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
            }
        }
        _ => (),
    });
}

fn get_framebuffers(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn get_render_command_buffers(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    vertex_buffer: &Arc<CpuAccessibleBuffer<[Vertex]>>,
    texture: &Arc<ImageView<StorageImage>>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            let layout = pipeline.layout().set_layouts().get(0).unwrap();
            let set = PersistentDescriptorSet::new(
                layout.clone(),
                [WriteDescriptorSet::image_view_sampler(
                    0,
                    texture.clone(),
                    Sampler::new(
                        device.clone(),
                        SamplerCreateInfo {
                            mag_filter: Filter::Linear,
                            min_filter: Filter::Linear,
                            address_mode: [SamplerAddressMode::Repeat; 3],
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                )],
            )
            .unwrap();

            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit, // don't forget to write the correct buffer usage
            )
            .unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassContents::Inline,
                )
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    set,
                )
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.read().unwrap().len() as u32, 1, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}

fn get_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Result<Arc<GraphicsPipeline>> {
    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .build(device)?;

    Ok(pipeline)
}

pub fn init_vulkan_and_surface(
    event_loop: &EventLoop<()>,
) -> Result<(Arc<Device>, Arc<Queue>, Arc<Surface<Window>>)> {
    let library = VulkanLibrary::new().expect("No local Vulkan library/DLL");
    let required_extensions = vulkano_win::required_extensions(&library);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .expect("Failed to create instance");

    let surface = WindowBuilder::new()
        .with_title("Vulkan Tutorial")
        .build_vk_surface(event_loop, instance.clone())?;

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..Default::default()
    };

    let (physical, queue_family_index) =
        select_physical_device(&instance, &surface, &device_extensions);

    info!("{:?}", physical.properties().device_name);
    info!(
        "{:?}",
        physical.supported_extensions().contains(&device_extensions)
    );

    let (device, mut queues) = Device::new(
        physical,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions,
            ..Default::default()
        },
    )?;

    let queue = queues.next().unwrap();

    Ok((device, queue, surface))
}

fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface<Window>>,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .filter(|p| p.supported_extensions().contains(device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                // Find the first first queue family that is suitable.
                // If none is found, `None` is returned to `filter_map`,
                // which disqualifies this physical device.
                .position(|(i, q)| {
                    q.queue_flags.graphics && p.surface_support(i as u32, surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,

            // Note that there exists `PhysicalDeviceType::Other`, however,
            // `PhysicalDeviceType` is a non-exhaustive enum. Thus, one should
            // match wildcard `_` to catch all unknown device types.
            _ => 4,
        })
        .expect("no device available")
}
