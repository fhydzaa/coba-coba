# ============================================================
# predictor.py
# ============================================================

import json
import re
import os

import numpy as np
import pandas as pd
import torch
import yaml
import sympy as sp

from kan import MultKAN
from kan.utils import SYMBOLIC_LIB


# ============================================================
# PYKAN CHECKPOINT LOADER
# ============================================================

def load_pykan_checkpoint(path="kan_numeric"):
    """
    Load checkpoint PyKAN yang dibuat oleh saveckpt().

    Checkpoint ini berasal dari model sendiri.
    """

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    with open(
        f"{path}_config.yml",
        "r"
    ) as stream:

        config = yaml.unsafe_load(stream)


    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    state = torch.load(
        f"{path}_state",
        map_location="cpu",
        weights_only=False
    )


    # --------------------------------------------------------
    # Reconstruct model
    # --------------------------------------------------------

    model = MultKAN(
        width=config["width"],
        grid=config["grid"],
        k=config["k"],
        mult_arity=config["mult_arity"],
        base_fun=config["base_fun_name"],
        symbolic_enabled=config["symbolic_enabled"],
        affine_trainable=config["affine_trainable"],
        grid_eps=config["grid_eps"],
        grid_range=config["grid_range"],
        sp_trainable=config["sp_trainable"],
        sb_trainable=config["sb_trainable"],
        state_id=config["state_id"],
        auto_save=config["auto_save"],
        first_init=False,
        ckpt_path=config["ckpt_path"],
        round=config["round"] + 1,
        device="cpu"
    )


    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(state)


    # --------------------------------------------------------
    # Cache data
    # --------------------------------------------------------

    model.cache_data = torch.load(
        f"{path}_cache_data",
        map_location="cpu",
        weights_only=False
    )


    # --------------------------------------------------------
    # Restore symbolic functions
    # --------------------------------------------------------

    depth = len(model.width) - 1

    for l in range(depth):

        out_dim = model.symbolic_fun[l].out_dim
        in_dim = model.symbolic_fun[l].in_dim

        funs_name = config[
            f"symbolic.funs_name.{l}"
        ]

        for j in range(out_dim):

            for i in range(in_dim):

                fun_name = funs_name[j][i]

                model.symbolic_fun[l].funs_name[j][i] = (
                    fun_name
                )

                model.symbolic_fun[l].funs[j][i] = (
                    SYMBOLIC_LIB[fun_name][0]
                )

                model.symbolic_fun[l].funs_sympy[j][i] = (
                    SYMBOLIC_LIB[fun_name][1]
                )

                model.symbolic_fun[l].funs_avoid_singularity[j][i] = (
                    SYMBOLIC_LIB[fun_name][3]
                )

    model.eval()

    return model


# ============================================================
# RUL PREDICTOR
# ============================================================

class RULPredictor:

    def __init__(self, model_dir="."):

        self.model_dir = model_dir


        # ----------------------------------------------------
        # Load configuration
        # ----------------------------------------------------

        with open(
            os.path.join(
                model_dir,
                "config.json"
            ),
            "r"
        ) as f:

            self.config = json.load(f)


        self.features = self.config["features"]
        self.target = self.config["target"]
        self.window = self.config["window"]

        self.input_id = self.config["input_id"]


        # ----------------------------------------------------
        # Load scaler
        # ----------------------------------------------------

        import joblib

        self.scaler_x = joblib.load(
            os.path.join(
                model_dir,
                "scaler_x.pkl"
            )
        )

        self.scaler_y = joblib.load(
            os.path.join(
                model_dir,
                "scaler_y.pkl"
            )
        )


        # ----------------------------------------------------
        # Load numeric KAN
        # ----------------------------------------------------

        self.kan = load_pykan_checkpoint(
            os.path.join(
                model_dir,
                "kan_numeric"
            )
        )


        # ----------------------------------------------------
        # Load symbolic formula
        # ----------------------------------------------------

        with open(
            os.path.join(
                model_dir,
                "symbolic_formula.txt"
            ),
            "r"
        ) as f:

            formula_text = f.read().strip()


        self.symbolic_formula = sp.sympify(
            formula_text
        )


        # ----------------------------------------------------
        # Sort x_1, x_2, ...
        # ----------------------------------------------------

        self.symbolic_symbols = sorted(
            self.symbolic_formula.free_symbols,
            key=self._symbol_number
        )


    @staticmethod
    def _symbol_number(symbol):

        match = re.search(
            r"(\d+)$",
            str(symbol)
        )

        if match:
            return int(match.group(1))

        return 999999


    # ========================================================
    # PREPARE INPUT
    # ========================================================

    def prepare_input(self, data):

        if isinstance(data, (dict, list)):
            data = pd.DataFrame(data)

        elif not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Input harus berupa DataFrame, dict, atau list."
            )

        data = data.copy()

        # ========================================================
        # VALIDASI 25 FEATURE
        # ========================================================

        missing_features = [
            feature
            for feature in self.features
            if feature not in data.columns
        ]

        if missing_features:
            raise ValueError(
                "Feature berikut tidak ditemukan: "
                + ", ".join(missing_features)
            )

        # ========================================================
        # VALIDASI CYCLE
        # ========================================================

        if "Discharge_cycle" not in data.columns:
            raise ValueError(
                "Kolom 'Discharge_cycle' wajib ada."
            )

        if len(data) < self.window:
            raise ValueError(
                f"Minimal membutuhkan {self.window} cycle."
            )

        # ========================================================
        # SORT + AMBIL 10 CYCLE TERAKHIR
        # ========================================================

        data = (
            data
            .sort_values("Discharge_cycle")
            .tail(self.window)
            .copy()
        )

        # ========================================================
        # 25 FEATURE
        # ========================================================

        X_raw = (
            data[self.features]
            .values
            .astype(np.float32)
        )

        # ========================================================
        # SCALING
        # ========================================================

        X_scaled = self.scaler_x.transform(
            X_raw
        )

        # ========================================================
        # FLATTEN
        # ========================================================

        X_flat = X_scaled.reshape(
            1, -1
        )

        return X_flat


    # ========================================================
    # KAN PREDICTION
    # ========================================================

    def predict_kan(
        self,
        X_flat
    ):

        # ----------------------------------------------------
        # Ambil 13 input hasil pruning
        # ----------------------------------------------------

        X_kept = X_flat[
            :,
            self.input_id
        ]


        # ----------------------------------------------------
        # PyKAN warning workaround
        # ----------------------------------------------------

        X_batch = np.repeat(
            X_kept,
            repeats=2,
            axis=0
        )


        X_tensor = torch.tensor(
            X_batch,
            dtype=torch.float32
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            pred_scaled = (
                self.kan(
                    X_tensor
                )
                .cpu()
                .numpy()
            )


        # Ambil sample pertama
        pred_scaled = pred_scaled[:1]


        # ----------------------------------------------------
        # Inverse target scaling
        # ----------------------------------------------------

        pred_rul = self.scaler_y.inverse_transform(
            pred_scaled
        )


        return float(
            pred_rul[0, 0]
        )


    # ========================================================
    # SYMBOLIC PREDICTION
    # ========================================================

    def predict_symbolic(
        self,
        X_flat
    ):

        # ----------------------------------------------------
        # Semua 13 input hasil pruning
        # ----------------------------------------------------

        X_kept = X_flat[
            0,
            self.input_id
        ]


        # ----------------------------------------------------
        # Ambil hanya simbol yang benar-benar
        # digunakan oleh formula
        # ----------------------------------------------------

        substitutions = {}

        for symbol in self.symbolic_symbols:

            symbol_text = str(symbol)

            match = re.search(
                r"(\d+)$",
                symbol_text
            )

            if match is None:
                raise ValueError(
                    f"Format simbol tidak dikenali: "
                    f"{symbol_text}"
                )

            # x_1 → index 0
            # x_2 → index 1
            # ...
            symbolic_index = (
                int(match.group(1)) - 1
            )


            # Pastikan index valid
            if (
                symbolic_index < 0
                or symbolic_index >= len(X_kept)
            ):

                raise ValueError(
                    f"Symbol {symbol_text} "
                    f"berada di luar input hasil pruning."
                )


            substitutions[symbol] = float(
                X_kept[symbolic_index]
            )


        # ----------------------------------------------------
        # Evaluasi formula
        # ----------------------------------------------------

        pred_scaled = float(
            self.symbolic_formula.evalf(
                subs=substitutions
            )
        )


        # ----------------------------------------------------
        # Inverse transform RUL
        # ----------------------------------------------------

        pred_scaled_array = np.array(
            [[pred_scaled]],
            dtype=np.float32
        )

        pred_rul = (
            self.scaler_y
            .inverse_transform(
                pred_scaled_array
            )
        )

        return float(
            pred_rul[0, 0]
        )


    # ========================================================
    # BOTH
    # ========================================================

    def predict(self, data):

        X_flat = self.prepare_input(
            data
        )


        kan_rul = self.predict_kan(
            X_flat
        )


        symbolic_rul = self.predict_symbolic(
            X_flat
        )


        return {
            "kan_rul": kan_rul,
            "symbolic_rul": symbolic_rul,
            "difference": abs(
                kan_rul - symbolic_rul
            )
        }