import math

import awkward
import dask
import dask_awkward
import numpy
import uproot


def _get_steps(
    normed_files,
    maybe_step_size=None,
    align_clusters=False,
    recalculate_seen_steps=False,
):
    nf_backend = awkward.backend(normed_files)
    lz_or_nf = awkward.typetracer.length_zero_if_typetracer(normed_files)

    array = [] if nf_backend != "typetracer" else lz_or_nf
    for arg in lz_or_nf:
        try:
            the_file = uproot.open({arg.file: None})
        except FileNotFoundError:
            array.append(None)
            continue

        tree = the_file[arg.object_path]
        num_entries = tree.num_entries

        target_step_size = num_entries if maybe_step_size is None else maybe_step_size

        file_uuid = str(the_file.file.uuid)

        out_uuid = arg.uuid
        out_steps = arg.steps

        if out_uuid != file_uuid or recalculate_seen_steps:
            if align_clusters:
                clusters = tree.common_entry_offsets()
                out = [0]
                for c in clusters:
                    if c >= out[-1] + target_step_size:
                        out.append(c)
                if clusters[-1] != out[-1]:
                    out.append(clusters[-1])
                out = numpy.array(out, dtype="int64")
                out = numpy.stack((out[:-1], out[1:]), axis=1)
            else:
                n_steps = math.ceil(num_entries / target_step_size)
                out = numpy.array(
                    [
                        [
                            i * target_step_size,
                            min((i + 1) * target_step_size, num_entries),
                        ]
                        for i in range(n_steps)
                    ],
                    dtype="int64",
                )

            out_uuid = file_uuid
            out_steps = out.tolist()

        array.append(
            {
                "file": arg.file,
                "object_path": arg.object_path,
                "steps": out_steps,
                "uuid": out_uuid,
            }
        )

    if len(array) == 0:
        array = awkward.Array(
            [
                {"file": "junk", "object_path": "junk", "steps": [[]], "uuid": "junk"},
                None,
            ]
        )
        array = awkward.Array(array.layout.form.length_zero_array(highlevel=False))
    else:
        array = awkward.Array(array)

    if nf_backend == "typetracer":
        array = awkward.Array(
            array.layout.to_typetracer(forget_length=True),
        )

    return array


def preprocess(
    fileset,
    maybe_step_size=None,
    align_clusters=False,
    recalculate_seen_steps=False,
    files_per_batch=1,
):
    out_updated = fileset.copy()
    out_available = fileset.copy()
    all_ak_norm_files = {}
    files_to_preprocess = {}
    for name, info in fileset.items():
        norm_files = uproot._util.regularize_files(info["files"], steps_allowed=True)
        for ifile in range(len(norm_files)):
            the_file_info = norm_files[ifile]
            maybe_finfo = info["files"].get(the_file_info[0], None)
            maybe_uuid = (
                None
                if not isinstance(maybe_finfo, dict)
                else maybe_finfo.get("uuid", None)
            )
            norm_files[ifile] += (3 - len(norm_files[ifile])) * (None,) + (maybe_uuid,)
        fields = ["file", "object_path", "steps", "uuid"]
        ak_norm_files = awkward.from_iter(norm_files)
        ak_norm_files = awkward.Array(
            {field: ak_norm_files[str(ifield)] for ifield, field in enumerate(fields)}
        )
        all_ak_norm_files[name] = ak_norm_files

        dak_norm_files = dask_awkward.from_awkward(
            ak_norm_files, math.ceil(len(ak_norm_files) / files_per_batch)
        )

        files_to_preprocess[name] = dask_awkward.map_partitions(
            _get_steps,
            dak_norm_files,
            maybe_step_size=maybe_step_size,
            align_clusters=align_clusters,
            recalculate_seen_steps=recalculate_seen_steps,
        )

    all_processed_files = dask.compute(files_to_preprocess)[0]

    for name, processed_files in all_processed_files.items():
        files_available = {
            item["file"]: {
                "object_path": item["object_path"],
                "steps": item["steps"],
                "uuid": item["uuid"],
            }
            for item in awkward.drop_none(processed_files).to_list()
        }

        files_out = {}
        for proc_item, orig_item in zip(
            processed_files.to_list(), all_ak_norm_files[name].to_list()
        ):
            item = orig_item if proc_item is None else proc_item
            files_out[item["file"]] = {
                "object_path": item["object_path"],
                "steps": item["steps"],
                "uuid": item["uuid"],
            }

        out_updated[name]["files"] = files_out
        out_available[name]["files"] = files_available

    return out_available, out_updated
