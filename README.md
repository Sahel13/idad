# Implicit Deep Adaptive Design (iDAD)

This is code for iDAD [1] modified for use in the paper "Nesting Particle Filters for Experimental Design in Dynamical Systems" [2].

To set up the Python environment and run experiments, follow the instructions at the [original repo](https://github.com/desi-ivanova/idad). The four experiments implemented for [2] are `pendulum_linear.py`, `pendulum.py`, `cartpole.py`, and `double_pendulum.py`. The sPCE lower bound for the trained models can be computed using `eval_sPCE.py`. To compute the EIG estimate for the trained models, run the respective file `{experiment_name}_eig.py`.

## References

[1] Ivanova, D. R., Foster, A., Kleinegesse, S., Gutmann, M., and Rainforth, T. Implicit deep adaptive design: Policy–based experimental design without likelihoods. In Advances in Neural Information Processing Systems. 2021. [Paper](https://arxiv.org/abs/2111.02329). [Code](https://github.com/desi-ivanova/idad).

[2] Iqbal, S., Corenflos, A., Särkkä, S., and Abdulsamad, H. Nesting Particle Filters for Experimental Design in Dynamical Systems. In International Conference on Machine Learning. 2024. [Paper](https://arxiv.org/abs/2402.07868). [Code](https://github.com/Sahel13/InsideOutSMC2.jl).
