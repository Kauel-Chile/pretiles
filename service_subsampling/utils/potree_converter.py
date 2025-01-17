import asyncio

async def run_potree_converter(input_file: str, output_file: str) -> bool:
    command = f"/home/opengeo/PotreeConverter/build/PotreeConverter {input_file} -o {output_file}"
    try:
        process = await asyncio.create_subprocess_shell(command)
        await process.communicate()
        
        return process.returncode == 0
    except Exception:
        return False