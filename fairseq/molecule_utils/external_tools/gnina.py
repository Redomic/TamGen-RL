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
            exhaustiveness: int = 8,
            num_modes: int = 9,
            seed: int = 1234,
            device: int = 0,
            cnn_scoring: str = "rescore",  # none, rescore, refinement, all
            scoring: str = "vina",  # ad4_scoring, dkoes_fast, vina, vinardo
            min_rmsd_filter: float = 1.0,
    ):
        if binary_path is None:
            binary_path = self.find_binary()
        if binary_path is None:
            raise RuntimeError('Must provide GNINA binary path.')
        self.binary_path = binary_path
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes
        self.seed = seed
        self.device = device
        self.cnn_scoring = cnn_scoring
        self.scoring = scoring
        self.min_rmsd_filter = min_rmsd_filter

    @staticmethod
    def find_binary() -> Optional[Path]:
        """Find gnina executable."""
        # Check common locations first
        common_paths = [
            Path("/home/redomic/Projects/TamGen-RL/gnina/build/bin/gnina"),
            Path("./gnina/build/bin/gnina"),
            Path("../gnina/build/bin/gnina"),
        ]
        
        for path in common_paths:
            if path.exists() and path.is_file():
                return path
        
        # Fall back to system PATH
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
        try:
            process = subprocess.Popen([str(self.binary_path), '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, _ = process.communicate()
            ret_code = process.wait()
            return ret_code == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _parse_output(self, output: str) -> float:
        """Parse GNINA output to extract binding affinity."""
        import re
        lines = output.splitlines()
        affinities = []
        
        for line in lines:
            # Look for lines with mode ranking and affinity
            # Format: "   1      -9.51      0.000"
            match = re.match(r"\s*\d+\s+(-?\d+\.\d+)", line)
            if match:
                try:
                    affinity = float(match.group(1))
                    affinities.append(affinity)
                except ValueError:
                    continue
        
        if not affinities:
            # Alternative parsing - look for "Affinity:" lines
            for line in lines:
                if "affinity" in line.lower():
                    numbers = re.findall(r'-?\d+\.\d+', line)
                    for num in numbers:
                        try:
                            affinity = float(num)
                            if -50.0 <= affinity <= 50.0:  # Reasonable range
                                affinities.append(affinity)
                        except ValueError:
                            continue
        
        if not affinities:
            raise GNINAError(f'Cannot find GNINA affinity scores in output:\n{output}')
        
        return min(affinities)  # Return best (most negative) affinity

    def _do_query(self, cmd):
        logging.info("Running GNINA command: %s", " ".join(str(x) for x in cmd))
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        with timing(f'GNINA query'):
            stdout, stderr = process.communicate()
            ret_code = process.wait()

        if ret_code != 0:
            logging.error("GNINA execution failed with return code %d", ret_code)
            logging.error("GNINA command: %s", " ".join(str(x) for x in cmd))
            logging.error("GNINA stderr:\n%s", stderr.decode("utf-8"))
            raise GNINAError(f'GNINA failed with return code {ret_code}\nstderr:\n{stderr.decode("utf-8")}\n')

        output = stdout.decode('utf-8')
        logging.debug("GNINA stdout:\n%s", output)
        
        return self._parse_output(output)

    def query(
            self, receptor_path: Path, ligand_path: Path, autobox_ligand_path: Optional[Path] = None,
            output_complex_path: Optional[Path] = None,
    ) -> float:
        """Run GNINA docking with autobox ligand or receptor."""
        if output_complex_path is None:
            output_complex_path = Path('/dev/null')

        cmd = [
            str(self.binary_path),
            '--receptor', str(receptor_path),
            '--ligand', str(ligand_path),
            '--exhaustiveness', str(self.exhaustiveness),
            '--num_modes', str(self.num_modes),
            '--seed', str(self.seed),
            '--device', str(self.device),
            '--cnn_scoring', str(self.cnn_scoring),
            '--scoring', str(self.scoring),
            '--min_rmsd_filter', str(self.min_rmsd_filter),
            '--out', str(output_complex_path),
        ]
        
        if autobox_ligand_path is not None:
            cmd.extend(['--autobox_ligand', str(autobox_ligand_path)])
        else:
            # If no autobox ligand, use the receptor for autobox
            cmd.extend(['--autobox_ligand', str(receptor_path)])

        return self._do_query(cmd)

    def query_box(
            self, receptor_path: Path, ligand_path: Path, center: Tuple[float, float, float],
            box: Tuple[float, float, float] = (20., 20., 20.), output_complex_path: Optional[Path] = None,
    ) -> float:
        """Run GNINA docking with specified box coordinates."""
        if output_complex_path is None:
            output_complex_path = Path('/dev/null')
            
        cmd = [
            str(self.binary_path),
            '--receptor', str(receptor_path),
            '--ligand', str(ligand_path),
            '--center_x', str(center[0]),
            '--center_y', str(center[1]), 
            '--center_z', str(center[2]),
            '--size_x', str(box[0]),
            '--size_y', str(box[1]),
            '--size_z', str(box[2]),
            '--exhaustiveness', str(self.exhaustiveness),
            '--num_modes', str(self.num_modes),
            '--seed', str(self.seed),
            '--device', str(self.device),
            '--cnn_scoring', str(self.cnn_scoring),
            '--scoring', str(self.scoring),
            '--min_rmsd_filter', str(self.min_rmsd_filter),
            '--out', str(output_complex_path),
        ]
        
        return self._do_query(cmd)