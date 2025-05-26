import contextlib
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple


@contextlib.contextmanager
def timing(msg: str):
    logging.info('Started %s', msg) 
    tic = time.time()
    yield
    toc = time.time()
    logging.info('Finished %s in %.3f seconds', msg, toc - tic)


class GNINAError(Exception):
    pass


class GNINA:
    def __init__(
            self, *,
            binary_path: Optional[Path] = None,
            exhaustiveness: int = 1,
            seed: int = 1234,
    ):
        if binary_path is None:
            binary_path = self.find_binary()
        if binary_path is None:
            raise RuntimeError('Must provide GNINA binary path.')
        self.binary_path = binary_path
        self.exhaustiveness = exhaustiveness
        self.seed = seed

    @staticmethod
    def find_binary() -> Optional[Path]:
        """Find gnina executable."""
        if os.name == 'nt':
            process = subprocess.Popen(['where.exe', 'gnina'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            process = subprocess.Popen(['which', 'gnina'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        ret_code = process.wait()

        if ret_code:
            return None

        return Path(stdout.decode().splitlines()[0].strip())

    def check_binary(self) -> bool:
        process = subprocess.Popen([str(self.binary_path), '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, _ = process.communicate()
        ret_code = process.wait()
        return not bool(ret_code)

    def _do_query(self, cmd):
        import re
        logging.info("Running GNINA command: %s", " ".join(cmd))
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        with timing(f'GNINA query'):
            stdout, stderr = process.communicate()
            ret_code = process.wait()

        if ret_code:
            logging.error("GNINA execution failed. Command: %s", " ".join(cmd))
            logging.error("GNINA stderr:\n%s", stderr.decode("utf-8"))
            raise GNINAError(f'GNINA failed\nstderr:\n{stderr.decode("utf-8")}\n')

        output = stdout.decode('utf-8')
        lines = output.splitlines()
        affinities = []
        for line in lines:
            match = re.match(r"\s*\d+\s+(-?\d+\.\d+)", line)
            if match:
                try:
                    affinities.append(float(match.group(1)))
                except ValueError:
                    continue
        if not affinities:
            raise GNINAError(f'Cannot find GNINA affinity scores in output:\n{output}')
        return min(affinities)

    def query(
            self, receptor_path: Path, ligand_path: Path, autobox_ligand_path: Optional[Path] = None,
            output_complex_path: Optional[Path] = None,
    ) -> float:
        if autobox_ligand_path is None:
            autobox_ligand_path = receptor_path
        if output_complex_path is None:
            output_complex_path = Path('/dev/null')

        cmd = [
            str(self.binary_path),
            '--receptor', str(receptor_path),
            '--ligand', str(ligand_path),
            '--autobox_ligand', str(autobox_ligand_path),
            '--exhaustiveness', str(self.exhaustiveness),
            '--seed', str(self.seed),
            '--out', str(output_complex_path),
        ]
        cmd.extend([
            '--num_modes', '1',
            '--device', '0',
        ])
        return self._do_query(cmd)

    def query_box(
            self, receptor_path: Path, ligand_path: Path, center: Tuple[float, float, float],
            box: Tuple[float, float, float] = (20., 20., 20.), output_complex_path: Optional[Path] = None,
    ) -> float:
        if output_complex_path is None:
            output_complex_path = Path('/dev/null')
        cmd = [
            str(self.binary_path),
            '--receptor', str(receptor_path),
            '--ligand', str(ligand_path),
            '--center_x', str(center[0]), '--center_y', str(center[1]), '--center_z', str(center[2]),
            '--size_x', str(box[0]), '--size_y', str(box[1]), '--size_z', str(box[2]),
            '--exhaustiveness', str(self.exhaustiveness),
            '--seed', str(self.seed),
            '--out', str(output_complex_path),
        ]
        cmd.extend([
            '--num_modes', '1',
            '--device', '0',
        ])
        return self._do_query(cmd)