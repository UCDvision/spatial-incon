# Dataset card

## Summary

The curated benchmark contains 615 image pairs from static indoor Hypersim scenes. Given a labeled reference image A and an edited image B, the task is to identify the one label in A whose object is spatially inconsistent with camera motion.

## Construction

Each example is based on three views of one scene and one object visible in all three. The object is removed from B, its surrounding region is inpainted, and the object appearance from the third view is inserted at B's original object location. The selected 615 pairs were manually inspected after automatic generation. `annotations.json` records source views, object id, answer letter, label count, and source artifacts.

`public_eval_annotations.json` adds evaluation-only grouping metadata: object depth, lighting, physical plausibility, label-count bucket, object class, and scene class.

## Intended use

Use the benchmark to evaluate multi-image spatial-consistency reasoning. The primary metric is exact answer-letter accuracy. A random-choice baseline must use the number of labels on each example rather than a fixed 26-way rate.

## Limitations

The data is synthetic and derived from indoor Hypersim scenes. It evaluates one kind of multiview inconsistency and is not a general measure of real-world 3D understanding. Image edits can retain subtle inpainting artifacts; the release includes scripts and recorded results for the paper's artifact controls.

## License

Released under the [MIT License](LICENSE).
