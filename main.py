from llamarpc.connection import LlamaRPCConnection
from llamarpc.types import RPCTensor, TensorType
from llamarpc.logger import logger, console
from rich.panel import Panel
from rich.table import Table

def main():
    conn = None
    try:
        console.print(Panel("[bold blue]Initializing LlamaRPC Connection", border_style="blue"))
        conn = LlamaRPCConnection("127.0.0.1", 50052)
        
        with console.status("[bold green]Testing buffer allocation..."):
            buffer_size = 1024
            buffer = conn.alloc_buffer(buffer_size)
            if buffer is None:
                console.print("[bold red]Failed to allocate buffer")
                return
            console.print(f"[bold green]Successfully allocated buffer at {hex(buffer)}")

        # Create a table for tensor details
        tensor_table = Table(title="Tensor Configuration")
        tensor_table.add_column("Property", style="cyan")
        tensor_table.add_column("Value", style="green")
        
        tensor = RPCTensor(
            id=1,
            type=TensorType.Q4_0,
            buffer=buffer.remote_ptr,
            ne=[1, 1, 1, 1],
            nb=[4, 4, 4, 4],
            op=0,
            op_params=[0] * 32,
            flags=0,
            src=[0] * 10,
            view_src=0,
            view_offs=0,
            data=0,
            name="test_tensor"
        )
        
        # Add tensor properties to table
        for field, value in tensor.__dict__.items():
            tensor_table.add_row(field, str(value))
        
        console.print(tensor_table)

        with console.status("[bold yellow]Processing operations...") as status:
            status.update("[bold blue]Setting tensor...")
            conn.set_tensor(tensor)
            console.print("[bold green]Successfully set tensor")

            status.update("[bold blue]Clearing buffer...")
            conn.buffer_clear(buffer.remote_ptr, 0)
            console.print("[bold green]Successfully cleared buffer")

            status.update("[bold blue]Freeing buffer...")
            conn.free_buffer(buffer.remote_ptr)
            console.print("[bold green]Successfully freed buffer")

    except Exception as e:
        console.print_exception()
    finally:
        if conn:
            conn.close()
        console.print("[bold blue]Connection closed")

if __name__ == "__main__":
    main()
