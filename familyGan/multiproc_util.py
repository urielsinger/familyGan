import multiprocessing
from tqdm.autonotebook import tqdm
from pathos.pools import ProcessPool as Pool

i = 0
proc_count = 1


def parmap(f, X, nprocs=multiprocessing.cpu_count(), force_parallel=False, chunk_size=1, use_tqdm=False, **tqdm_kwargs):

    if len(X) == 0:
        return []  # like map

    # nprocs = min(nprocs, cn.max_procs)
    if nprocs != multiprocessing.cpu_count() and len(X) < nprocs * chunk_size:
        chunk_size = 1  # use chunk_size = 1 if there is enough procs for a batch size of 1
    nprocs  = int(max(1, min(nprocs, len(X) / chunk_size)))  # at least 1
    if len(X) < nprocs:
        if nprocs != multiprocessing.cpu_count(): print("parmap too much procs")
        nprocs = len(X)  # too much procs

    if nprocs == 1 and not force_parallel:  # we want it serial (maybe for profiling)
        return list(map(f, X))

    def _spawn_fun(input, func):
        import random, numpy
        random.seed(1554+i); numpy.random.seed(42+i)  # set random seeds
        try:
            res = func(input)
            res_dict = dict()
            res_dict["res"] = res
            # res_dict["functions_dict"] = function_cache2.caches_dicts
            # res_dict["experiment_purpose"] = cn2.experiment_purpose
            # res_dict["curr_params_list"] = cn2.curr_experiment_params_list
            return res_dict
        except:
            import traceback
            traceback.print_exc()
            raise  # re-raise exception

    # if chunk_size == 1:
    #     chunk_size = math.ceil(float(len(X)) / nprocs)  # all procs work on an equal chunk

    try:  # try-catch hides bugs
        global proc_count
        old_proc_count = proc_count
        proc_count = nprocs
        p = Pool(nprocs)
        p.restart(force=True)
        # can throw if current proc is daemon
        if use_tqdm:
            retval_par = tqdm(p.imap(_spawn_fun, X, [f] * len(X), chunk_size=chunk_size), total=len(X), **tqdm_kwargs)
        else:
            retval_par = p.map(_spawn_fun, X, [f]*len(X), chunk_size=chunk_size)

        retval = list(map(lambda res_dict: res_dict["res"], retval_par))  # make it like the original map

        p.terminate()
        # for res_dict in retval_par:  # add all experiments params we missed
        #     curr_params_list = res_dict["curr_params_list"]
        #     for param in curr_params_list:
        #         cn.add_experiment_param(param)
        # cn.experiment_purpose = retval_par[0]["experiment_purpose"]  # use the "experiment_purpose" from the fork
        # function_cache.merge_cache_dicts_from_parallel_runs(map(lambda a: a["functions_dict"], retval_par))  # merge all
        proc_count = old_proc_count
        global i
        i += 1
    except AssertionError as e:
        if e.message == "daemonic processes are not allowed to have children":
            retval = map(f, X)  # can't have pool inside pool
        else:
            print("error message is: " + str(e.message))
            raise  # re-raise orig exception
    return retval
