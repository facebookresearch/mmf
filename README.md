
<div align="center">
<img src="https://mmf.sh/img/logo.svg" width="50%"/>
</div>

#

<div align="center">
  <a href="https://mmf.sh/docs">
  <img alt="Documentation Status" src="https://readthedocs.org/projects/mmf/badge/?version=latest"/>
  </a>
  <a href="https://circleci.com/gh/facebookresearch/mmf">
  <img alt="CircleCI" src="https://circleci.com/gh/facebookresearch/mmf.svg?style=svg"/>
  </a>
</div>

---

MMF is a modular framework for vision and language multimodal research from Facebook AI Research. MMF contains reference implementations of state-of-the-art vision and language models and has powered multiple research projects at Facebook AI Research. See full list of project inside or built on MMF [here](https://mmf.sh/docs/notes/projects).

MMF is powered by PyTorch, allows distributed training and is un-opinionated, scalable and fast. Use MMF to **_bootstrap_** for your next vision and language multimodal research project by following the [installation instructions](https://mmf.sh/docs/getting_started/installation). Take a look at list of MMF features [here](https://mmf.sh/docs/getting_started/features).

MMF also acts as **starter codebase** for challenges around vision and
language datasets (The Hateful Memes, TextVQA, TextCaps and VQA challenges). MMF was formerly known as Pythia. The next video shows an overview of how datasets and models work inside MMF. Checkout MMF's [video overview](https://mmf.sh/docs/getting_started/video_overview).


## Installation

Follow installation instructions in the [documentation](https://mmf.sh/docs#creating-a-conda-environment-optional).

## Documentation

Learn more about MMF [here](https://mmf.sh/docs).

### Segmentation Fault ###
It generally occurs when the program is trying to read or write an illegal memory location, for example, trying to read or write to a non-existent array element, not properly defined pointers etc.
It can be handled using the following utilities:-
* #### htop ####
  It allows us to monitor running processes on the system, their memory usage, CPU usage etc
  Here, looking into VIRT (Virtual memory usage), RES (Physical RAM usage in kb), SHR (Shared memory), PID (Process ID)   will give insights into the problem.
* #### vmstat ####
  Segfault's common reason is to access the part of the virtual address space that is not mapped to a physical one.
  vmstat (virtual memory statistics) might be helpful here, through this system activity can be observed in real-time.


## Citation

If you use MMF in your work or use any models published in MMF, please cite:

```bibtex
@misc{singh2020mmf,
  author =       {Singh, Amanpreet and Goswami, Vedanuj and Natarajan, Vivek and Jiang, Yu and Chen, Xinlei and Shah, Meet and
                 Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
  title =        {MMF: A multimodal framework for vision and language research},
  howpublished = {\url{https://github.com/facebookresearch/mmf}},
  year =         {2020}
}
```

## License

MMF is licensed under BSD license available in [LICENSE](LICENSE) file
