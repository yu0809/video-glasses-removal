# input/

Put your videos here. `example.mp4` is the only file shipped with the repository:

- **Source**: the driving-video sample `assets/examples/driving/d9.mp4` of
  [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait) (MIT licence), unchanged.
- It is used for the quick test in the README and for the real-video figure in the blog post.

```bash
python remove_glasses.py -i input/example.mp4 --duration 2
```
