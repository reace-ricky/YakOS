
        # -- Auto-write fades to bias -------------------------------------------
        from yak_core.bias import load_bias, save_bias

        bias = load_bias()
        manual_fades = [n for n, v in bias.items() if v.get("max_exposure", 1.0) == 0.0]

        # Original logic of _render_the_board continues here...
