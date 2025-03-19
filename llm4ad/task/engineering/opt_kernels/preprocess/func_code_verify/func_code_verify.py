import os
import time
import torch
import tempfile

from ..code_verify import CodeVerify

class FuncCodeVerify(CodeVerify):
    def __init__(self, verify_counts=5, res_path=None, keep_temp=False):
        super().__init__(verify_counts=verify_counts, res_path=res_path, keep_temp=keep_temp, specifier="FuncVerify")

    def verify_func_code(self, org_code: str, func_code: [None, str], device, seed=0) -> (bool, str):
        assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
        assert org_code is not None and func_code is not None
        torch.set_printoptions(
            precision=4,  # Decimal places
            threshold=10,  # Total number of elements before truncating
            edgeitems=3,  # Number of elements at beginning and end of dimensions
            linewidth=80,  # Maximum width before wrapping
        )
        # set CUDA device
        torch.cuda.set_device(device)

        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        temp_dir = tempfile.TemporaryDirectory(dir=self.res_path, prefix=f"{time_stamp}_", delete=False)
        os.makedirs(temp_dir.name, exist_ok=True)
        org_code_path = os.path.join(temp_dir.name, "original.py")
        func_code_path = os.path.join(temp_dir.name, "func.py")
        self.write_multiple_files_to_multiple_paths([org_code_path, func_code_path], [org_code, func_code])

        org_module, org_spec = self.load_module_from_path(org_code_path, "org_module")
        try:
            func_module, func_spec = self.load_module_from_path(func_code_path, "func_module")
        except Exception as e:
            if not self.keep_temp:
                self.clear_dir(temp_dir.name)
            return False, f"Syntax error for the functional code: {str(e)}"


        self.set_seed(seed)
        init_inputs = org_module.get_init_inputs()
        init_inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]
        func_init_inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]

        with torch.no_grad():
            self.set_seed(seed)
            org_model_inst = org_module.Model(*init_inputs)
            assert hasattr(org_model_inst, "forward")
            torch.cuda.synchronize(device=device)
            try:
                func_model_inst = func_module.Model(*func_init_inputs)
                assert hasattr(org_model_inst, "forward")
                torch.cuda.synchronize(device=device)
            except Exception as e:
                if not self.keep_temp:
                    self.clear_dir(temp_dir.name)
                return False, f"Failed to create the model from the functional code: {str(e)}"

        return self.check_correctness(org_model_inst, func_model_inst, org_module.get_inputs, device, seed)



    def check_correctness(self, org_model_inst, func_model_inst, get_inputs, device, seed, atol=1e-02, rtol=1e-02):
        pass_count = 0

        torch.manual_seed(seed)
        correctness_trial_seeds = [
            torch.randint(0, 2 ** 32 - 1, (1,)).item() for _ in range(self.verify_counts)
        ]

        with torch.no_grad():
            for trial in range(self.verify_counts):
                trial_seed = correctness_trial_seeds[trial]
                self.set_seed(trial_seed)
                inputs = get_inputs()
                inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                self.set_seed(trial_seed)
                model = org_model_inst.cuda(device=device)
                self.set_seed(trial_seed)
                func_model = func_model_inst.cuda(device=device)

                output = model(*inputs)
                torch.cuda.synchronize(device=device)

                try:
                    output_new = func_model(*inputs)
                    torch.cuda.synchronize(device=device)
                    if output.shape != output_new.shape:
                        return False, f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                    if not torch.allclose(
                            output, output_new, atol=atol, rtol=rtol
                    ):  # fail
                        max_diff = torch.max(torch.abs(output - output_new)).item()
                        avg_diff = torch.mean(torch.abs(output - output_new)).item()
                        return False, f"Output mismatch: max_diff={max_diff:.6f}, avg_diff={avg_diff:.6f}"
                    else:
                        pass_count += 1
                except Exception as e:
                    return False, f"Error running the functional model: {str(e)}"
        if pass_count == self.verify_counts:
            return True, None
        else:
            return False, f"Failed {self.verify_counts-pass_count}/{self.verify_counts} trials of correctness check."