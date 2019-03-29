import numpy as np
import cv2
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def calc_distance_between_2_vectors(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dist = np.sqrt(np.sum(np.square(v1 - v2)))
    return dist


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def get_all_detected_boxes(
        original_boxes,
        scores,
        max_boxes_to_draw=20,
        min_score_thresh=0.6):
    boxes = []
    original_boxes = np.squeeze(original_boxes)
    scores = np.squeeze(scores)
    for i in range(min(max_boxes_to_draw, original_boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = original_boxes[i].tolist()
            boxes.append(box)
    return boxes


def get_point(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def get_width(box):
    return abs(box[3] - box[1])


def get_player_img(box, image_np, small=False):
    im_width, im_height, _ = image_np.shape
    ymin, xmin, ymax, xmax = box
    player_img = image_np[int(im_width * ymin): int(im_width * ymax) + 1,
                          int(im_height * xmin): int(im_height * xmax) + 1]
    if small:
        return cv2.resize(player_img, (28, 28))
    return cv2.resize(player_img, (128, 128))

def visualize_tracklets_on_image_array(image_np,
                                       tracklets,
                                       use_normalized_coordinates,
                                       line_thickness):
  image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
  for tracklet in tracklets:
      visualize_tracklet_on_image(image_pil, tracklet,
                                  use_normalized_coordinates,
                                  line_thickness)
      draw_bounding_box_on_image(image_pil,
                                 tracklet.detections[-1].box,
                                 ['player:' + str(tracklet.id)],
                                 'blue',
                                 2,
                                 use_normalized_coordinates)

  np.copyto(image_np, np.array(image_pil))


def visualize_tracklet_on_image(image_pil,
                                  tracklet,
                                  use_normalized_coordinates,
                                  line_thickness):
    draw = ImageDraw.Draw(image_pil)
    im_width, im_height = image_pil.size
    if not use_normalized_coordinates:
        im_width = 1
        im_height = 1
    draw.line([(x[1]*im_width, x[0]*im_height) for x in tracklet.get_points()],
              fill=get_color(tracklet),
              width=line_thickness)


def draw_bounding_box_on_image(image,
                                 bbox,
                                 display_str_list,
                                 color='blue',
                                 thickness=4,
                                 use_normalized_coordinates=True):
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  ymin, xmin, ymax, xmax = bbox
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('helper/arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin



def get_color(tracklet):
  return STANDARD_COLORS[int(tracklet.color / 30.0 * len(STANDARD_COLORS))]