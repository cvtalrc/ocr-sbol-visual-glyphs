import numpy as np
import cv2
import torch
import glob as glob
import matplotlib.pyplot as plt

from config import CLASSES, NUM_CLASSES
from model import create_model

# set the computation device
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load the model and the trained weights
model = create_model(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(
    '../outputs/model150.pth', map_location=device
))
model.eval()

# directory where all the images are present
DIR_TEST = '../test_data'
test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test instances: {len(test_images)}")

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.1

for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split('/')[-1].split('.')[0]
    print(image_name)
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    
    image = np.transpose(image, (2, 0, 1)).astype(np.float64)

    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cpu()
    # add batch dimension
    image = torch.unsqueeze(image, 0)

    with torch.no_grad():
        outputs = model(image)
    
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    print(len(outputs[0]['boxes']))

    # if len(outputs[0]['boxes']) != 0:
    #     boxes = outputs[0]['boxes'].data.numpy()
    #     scores = outputs[0]['scores'].data.numpy()
    #     # filter out boxes according to `detection_threshold`
    #     boxes = boxes[scores >= detection_threshold].astype(np.int32)
    #     draw_boxes = boxes.copy()
    #     # get all the predicited class names
    #     pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
    #     print(scores)
    #     print(pred_classes)
        
    #     # draw the bounding boxes and write the class name on top of it
    #     for j, box in enumerate(draw_boxes):
    #         line_thickness = 1 

    #         cv2.rectangle(orig_image,
    #                     (int(box[0]), int(box[1])),
    #                     (int(box[2]), int(box[3])),
    #                     (0, 0, 255), line_thickness)

    #         font_scale = 0.5 
    #         font_thickness = 1  

    #         text_position = (int(box[0]), int(box[1]) - 10)

    #         cv2.putText(orig_image, pred_classes[j], 
    #                     text_position,
    #                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 
    #                     font_thickness, lineType=cv2.LINE_AA)

    #     cv2.imshow('Prediction', orig_image)
    #     cv2.waitKey(1)
    #     cv2.imwrite(f"../test_predictions/{image_name}.jpg", orig_image,)

    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)

        # Encontrar el índice de la caja con el score más alto
        max_score_index = np.argmax(scores)

        # Seleccionar solo la caja con el score más alto
        draw_box = boxes[max_score_index]
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        print(scores)
        print(pred_classes)
        # Obtener la clase predicha y el score para la caja seleccionada
        pred_class = CLASSES[outputs[0]['labels'].cpu().numpy()[max_score_index]]
        pred_score = scores[max_score_index]

        # Dibujar la caja y escribir el texto
        line_thickness = 1 
        cv2.rectangle(orig_image,
                    (int(draw_box[0]), int(draw_box[1])),
                    (int(draw_box[2]), int(draw_box[3])),
                    (0, 0, 255), line_thickness)

        font_scale = 0.5 
        font_thickness = 1  

        text_position = (int(draw_box[0]), int(draw_box[1]) - 10)

        cv2.putText(orig_image, f"{pred_class} {pred_score:.2f}", 
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 
                    font_thickness, lineType=cv2.LINE_AA)

        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(1)
        cv2.imwrite(f"../test_predictions/{image_name}.jpg", orig_image)

    print(f"Image {i+1} done...")
    print('-'*50)

print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()