

use image::{ImageBuffer, Rgba};
use log::{info};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo,
};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};

use vulkano::format::{Format};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageDimensions, StorageImage};

use vulkano::pipeline::PipelineBindPoint;
use vulkano::pipeline::{ComputePipeline, Pipeline};
use vulkano::sync::GpuFuture;
use vulkano::{sync};





use anyhow::Result;

use vulkan::init_vulkan;

mod cs {
    use vulkano_shaders::shader;

    shader!(ty: "compute", path: "shaders/mandelbrot.glsl");
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    let (device, queue) = init_vulkan()?;

    let image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: 8192,
            height: 8192,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(queue.queue_family_index()),
    )?;

    let result_buf = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage {
            transfer_dst: true,
            ..Default::default()
        },
        false,
        (0..8192 * 8192 * 4).map(|_| 0u8),
    )?;

    let shader = cs::load(device.clone())?;

    let pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )?;

    let image_view = ImageView::new_default(image.clone())?;

    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::image_view(0, image_view)],
    )?;

    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;
    builder
        .bind_pipeline_compute(pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set,
        )
        .dispatch([8192 / 8, 8192 / 8, 1])?
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image,
            result_buf.clone(),
        ))?;

    let command_buffer = builder.build()?;

    sync::now(device)
        .then_execute(queue, command_buffer)?
        .then_signal_fence_and_flush()?
        .wait(None)?;

    let buffer_content = result_buf.read()?;
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(8192, 8192, &buffer_content[..]).unwrap();

    image.save("image.png")?;

    info!("Success !");

    Ok(())
}
