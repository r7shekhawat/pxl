use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "pxl", about = "A fast batch image processor", version)]
pub struct Args {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Resize images to a target width
    Resize {
        /// Input file or directory
        input: String,
        /// Output directory
        out: String,
        /// Target width in pixels
        #[arg(long)]
        width: u32,
    },
    /// Convert images to another format
    Convert {
        /// Input file or directory
        input: String,
        /// Target format (png, jpeg, webp)
        #[arg(long)]
        to: String,
    },
}

pub fn parse() -> Args {
    Args::parse()
}
