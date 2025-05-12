# 🤖 KGInPaint: InPainting with Interactive Scene Graphs

This readme serves as a manual for setting up the RelTR Scene Graph Generation model and then setting up the KGInPaint dashboard for inpainting within an input image using the generated scene graph for a given image. It is advised that you run KGInPaint on a device with a CUDA compatible GPU.

---

## 🔍 Introduction

Conventional image editing often requires laborious pixel-level work for complex modifications. KGInPaint addresses this by combining scene-graph generation with interactive inpainting, allowing users to semantically manipulate objects within an image through a simple dashboard interface.

---

## 🎥 Demo Video

Please click on the Video Thumbnail below to watch the demo for our KGInPaint Dashboard.

[![KGInPaint Demo Video Link](./images/demo.png)](https://www.youtube.com/watch?v=tGeAod6Y3TA)

---

## ✨ Key Features

* **Interactive Scene Graph Generation**: Uses RelTR to predict a structured scene graph from any input image.
* **Node & Edge Selection**: Click or drag to select objects and relationships directly on the graph; bounding boxes appear on the image.
* **Object Removal**: Select a node and press "Remove Object" to inpaint the background automatically.
* **Object Replacement**: Select a node, enter a text prompt (e.g., "add green tree"), and press "Replace Object" to generate multiple inpainted variants.
* **Real-time Preview**: Carousel displays inpainted outputs for quick comparison and download.

---

## ⚙️ How It Works

1. **Scene Graph Generation**: RelTR encodes the image and predicts object–predicate–object triplets.
2. **User Interaction**: Dashboard lets users select graph nodes or edges; highlights the corresponding regions on the image.
3. **Segmentation**: SAM model produces a precise mask of the selected object.
4. **Inpainting**: Hugging Face Stable Diffusion Inpainting v2 takes the mask and optional prompt to generate new image variants.

---

## 📊 User Survey Results

| Area                                             | Average Rating (out of 5) |
| ------------------------------------------------ | ------------------------- |
| Usage and First Impressions                      | 4.7                       |
| Scene Graph Generation Quality                   | 3.7                       |
| Inpainting Quality                               | 4.2                       |
| Effectiveness of Interactivity and Response Time | 4.5                       |
| Overall Satisfaction                             | 4.2                       |

---

## 🛠️ Virtual Environment & Requirements

If you do not have Python virtualenv installed, please run the following command to install virtualenv:

```bash
pip install virtualenv  # or pip3 install virtualenv or python3 -m pip install virtualenv
```

Setup the virtualenv by running the following commands:

```bash
python3 -m virtualenv .kginpaint_env
source .kginpaint_env/bin/activate
```

Alternatively you can setup a Conda Environment by installing conda by following this [link](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). (Recommended)

```bash
conda create -n kginpaint_env python=3.11.8
conda activate kginpaint_env
```

Install the necessary requirements:

```bash
pip install -r requirements.txt
```

---

## ⚙️ RelTR Setup

Download the [RelTR model](https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD/view) pretrained on the Visual Genome dataset and put it under:

```bash
./sgg_model/ckpt/checkpoint0149.pth
```

The RelTR codebase is included here for convenience. For the original implementation, visit the [RelTR GitHub repository](https://github.com/yrcong/RelTR).

---

## 📁 Input Images Setup

Move all your input images to the folder `./data/images`:

```bash
mv [your_curr_directory]/[your_input_image].jpg [repository_directory]/KGInPaint/data/images
```

---

## 🚀 Using the KGInPaint Dashboard

Host the application (with your virtual environment activated):

```bash
python3 kginpaint_app.py
```

Open your browser to `http://localhost:8050/`. Drag & drop or browse to load an image.

### 🖼️ Scene Graph Manipulation Features

* Select nodes/edges on the graph to highlight regions on the image.
* Use dropdown to color-code relationships.
* Zoom and pan both the graph and image.

---

### 🎨 Inpainting Features

* **Remove Object**: Select a graph node, then click 'Remove Object' to inpaint the background.
* **Replace Object**: Select a node, enter a prompt in the text box next to 'Replace Object', and click to generate variants. Outputs appear in a carousel—right-click to save.

---

## ✍️ Author

* Aniruddha Prashant Deshpande [GitHub Profile](https://github.com/aniruddhapdeshpande99)

---

## 📚 References

```bibtex
@article{cong2023reltr,
  title={Reltr: Relation transformer for scene graph generation},
  author={Cong, Yuren and Yang, Michael Ying and Rosenhahn, Bodo},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```
