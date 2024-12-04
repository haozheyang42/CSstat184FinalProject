from isolation_env import env
from pettingzoo.test import api_test, render_test

if __name__ == "__main__":
    _env = env()
    api_test(_env, num_cycles=1000, verbose_progress=False)

    env_func = env
    render_test(env_func)
