mod cli;
mod ops;
mod pipeline;

use anyhow::Result;

fn main() -> Result<()> {
    let args = cli::parse();
    match args.command {
        cli::Commands::Resize { input, out, width } => {
            pipeline::run_resize(&input, &out, width)
        }
        cli::Commands::Convert { input, to } => {
            pipeline::run_convert(&input, to)
        }
    }
}
