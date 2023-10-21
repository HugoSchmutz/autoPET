import colorsys
import random
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.patches import Polygon
from skimage.measure import find_contours

seed_color = 0


def random_colors(N: int, bright: Optional[bool]=True) -> List[Tuple]:
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.

    Notes
    -----
    https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.Random(seed_color).shuffle(colors)
    return colors


def apply_mask(image: np.array, mask: np.array, color: Sequence[float], alpha: Optional[float]=0.5) -> np.array:
    """
    Apply the given mask to the image.

    Notes
    -----
    https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
    """
    for c in range(3):
        image[:, :, c] = np.where(mask > 0.5,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def get_bbox(mask: np.array) -> np.array:
    """Generate the bounding box of each instance from mask

    Parameters
    ----------
    mask : ndarray
        instance segmentation: [height, width, num_instances]

    Returns
    -------
    ndarray
        the bounding box: [num_instances, (y1, x1, y2, x2)]
    """
    bbox = []
    for i in range(mask.shape[2]):
        indexes = np.where(mask[:, :, i])
        if len(indexes[0]) == 0:
            y1, y2 = 0, 1
            x1, x2 = 0, 1
        else:
            y1, y2 = min(indexes[0]), max(indexes[0])
            x1, x2 = min(indexes[1]), max(indexes[1])
        bbox.append([y1, x1, y2, x2])

    bbox = np.array(bbox)
    return bbox


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object

    Notes
    -----
    https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        # ax.text(x1, y1 + 8, caption,
        #         color='w', size=11, backgroundcolor="none")
        # ax.text(x2, y1, caption,
        #         color=color, size=11, backgroundcolor="none") 
        # x_caption = int(x1/2) if x1+x2 < width else int((x2 + width - len(caption))/2)
        ax.text(width - len(caption), y1, caption,
                color=color, size=11, backgroundcolor="none") 

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()


def get_src_image(pet_array: np.array) -> np.array:
    """Transform 2D PET scan (MIP for ex.) into RGB image.

    Parameters
    ----------
    pet_array : ndarray
        PET scan (SUV) of shape (height, width)

    Returns
    -------
    ndarray
        RGB PET of shape (height, width, 3)
    """
    image = pet_array.copy()

    # Normalize
    suv_clip = 10
    image = np.clip(image, a_max=suv_clip, a_min=0)  # Normalize to 0-1 (float)
    image = (255 * image / suv_clip).astype(int)  # Normalize to 0-255 (int)
    image = 255 - image # low SUV = white, high SUV black

    # transform to RGB
    image = image[:, :, None] * np.ones(3, dtype=int)[None, None, :]
    return image


def get_colors(labels: List[str]):
    """return color to use for each label. 
    Same label will have the same color but 2 differents labels could have the same color.

    Parameters
    ----------
    labels : list[str]
        list of label, for ex: ['Lung', 'brain', 'bone']

    Returns
    -------
    list
        list of color to use
    """
    max_colors = 255
    c = random_colors(max_colors)
    colors = [c[int.from_bytes(txt.encode(), 'little') % max_colors] for txt in labels]
    return colors


def show_mip_pet_and_mask(pet_array: np.array, mask_array: np.array, axis: Optional[int]=1, 
                          labels: Optional[List[str]]=None, show_labels: Optional[bool]=False, show_bbox: Optional[bool]=False, 
                          *args,**kwargs): # ax=None
    """Display MIP of PET scan and mask on the same plot
    PET scan and mask must be aligned

    Parameters
    ----------
    pet_array : ndarray
        PET scan (SUV) of shape (z, y, x)
    mask_array : ndarray or None
        semantic segmentation of shape (z, y, x) or
        instance segmentation of shape (z, y, x, num_instances)
        If set to None, only PET is displayed
    axis : int, optional
        axis to perform the MIP
        0 for axial, 1 for coronal, 2 for sagittal, by default 1
    ax : matplotlib.axes.Axe, optional
        Matplotlib axis to draw on. If set to None, one will be created.
        by default None
    show_bbox : bool, optional
        To show masks and bounding boxes or not, by default False
    labels: list or None, [num_instances]
        labels[i] = label of mask_array[:, :, :, i]

    Examples
    -----
    fig, axes = plt.subplots(1, 2, figsize=(20, 20))
    img_dict = {'pet': sitk_pet_nifti,
                'mask': sitk_mask_nifti}

    show_mip_pet_and_mask(pet_array=sitk.GetArrayFromImage(img_dict['pet']), 
                        mask_array=None, axis=1,
                        ax=axes[0])

    show_mip_pet_and_mask(pet_array=sitk.GetArrayFromImage(img_dict['pet']), 
                        mask_array=sitk.GetArrayFromImage(img_dict['mask']), axis=1,
                        ax=axes[1])

    fig.tight_layout()
    plt.show()
    """
    # reverse z axis => head up
    # numpy shape = (z, y x), nifti/sitk shape = (x, y, z)
    #pet_array = np.flip(pet_array, axis=0)
    #mask_array = np.flip(mask_array, axis=0) if mask_array is not None else np.zeros(pet_array.shape)

    # apply MIP
    print(pet_array.shape, axis)
    pet_array = np.max(pet_array, axis=axis)
    mask_array = np.max(mask_array, axis=axis)
    
    # convert to RGB
    image = get_src_image(pet_array) # image.shape = (height, width, 3)

    # (height, width) to (height, width, num_instances=1)
    if len(mask_array.shape) == 2:
            mask_array = np.expand_dims(mask_array, axis=-1) 

    # bbox.shape = [num_instances, 4]
    bbox = get_bbox(mask_array) if np.count_nonzero(mask_array) != 0 else np.array([[0, 0, 0, 0]])

    if labels is None:
        # generate dummy info
        class_ids, class_names = np.ones(bbox.shape[0], dtype=int), ["", ""]
        colors = None
    else:
        assert len(labels) == bbox.shape[0]
        class_ids = np.arange(bbox.shape[0], dtype=int)
        class_names = labels if show_labels else [""] * len(labels)
        colors = get_colors(labels)

    display_instances(image, bbox, mask_array, class_ids, class_names, 
                      colors=colors, show_bbox=show_bbox,
                      *args, **kwargs) # ax=ax


def plot_diff(pet_array: np.array, gt_mask_array: np.array, pred_mask_array: np.array, 
              axis: Optional[int]=1, *args,**kwargs):
    """Fonction to prepapre input before plotting the difference between pred and ground-truth.

    Parameters
    ----------
    pet_array : ndarray
        PET scan (SUV) of shape (z, y, x)
    gt_mask_array : ndarray
        binary ground-truth of the semantic segmentation of shape (z, y, x)
    pred_mask_array : ndarray
        binary prediction of the semantic segmentation of shape (z, y, x)
    axis : int, optional
        axis to perform the MIP
        0 for axial, 1 for coronal, 2 for sagittal, by default 1
    ax : matplotlib.axes.Axe, optional
        Matplotlib axis to draw on. If set to None, one will be created.
        by default None
    """

    # flip so the head is at the top of the image
    pet_array = np.flip(pet_array, axis=0)
    gt_mask_array = np.flip(gt_mask_array, axis=0)
    pred_mask_array = np.flip(pred_mask_array, axis=0)

    # mip
    pet_array = np.max(pet_array, axis=axis)
    gt_mask_array = np.max(gt_mask_array, axis=axis)
    pred_mask_array = np.max(pred_mask_array, axis=axis)

    # background of the plot 
    image = get_src_image(pet_array) # convert to RGB

    # Add dim to create semantic segmentation with 1 instance
    gt_mask_array = np.expand_dims(gt_mask_array, axis=-1)
    pred_mask_array = np.expand_dims(pred_mask_array, axis=-1)

    plot_difference(image, gt_mask_array, pred_mask_array, *args,**kwargs)


def plot_difference(image: np.array, gt_mask: np.array, pred_mask: np.array, 
                    title="Ground Truth and Detections\n GT=green, pred=red", *args,**kwargs):
    """Display ground truth and prediction on the same image.

    Parameters
    ----------
    image : ndarray
        background image (usually the MIP-PET scan) of shape (y, x)
    gt_mask : ndarray
        binary ground-truth of the semantic segmentation of shape (y, x, 1)
    pred_mask : ndarray
        binary prediction of the semantic segmentation of shape (y, x, 1)
    title: str
        title of the subplot
    ax : matplotlib.axes.Axe, optional
        Matplotlib axis to draw on. If set to None, one will be created.
        by default None
    """
    # Get bounding box for each mask
    gt_bbox = get_bbox(gt_mask)
    pred_bbox = get_bbox(pred_mask)

    # generate things
    gt_colors = [(0, 1, 0, .8)] * gt_mask.shape[2]  # Ground truth = green.
    gt_class_ids = np.ones(gt_mask.shape[2], dtype=int)

    pred_colors = [(1, 0, 0, 1)] * pred_mask.shape[2]  # Predictions = red
    pred_class_ids = np.ones(pred_mask.shape[2], dtype=int)

    # Concatenate GT and predictions
    class_ids = np.concatenate([gt_class_ids, pred_class_ids])
    class_names = ["", ""]
    colors = gt_colors + pred_colors
    boxes = np.concatenate([gt_bbox, pred_bbox])
    masks = np.concatenate([gt_mask, pred_mask], axis=-1)

    # Display
    display_instances(
        image,
        boxes, masks, class_ids,
        class_names,
        show_bbox=False, show_mask=True,
        colors=colors,
        title=title,
        *args,**kwargs)
