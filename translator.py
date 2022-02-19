import cv2

def translate(dataset):
    
    img, (rel_questions, rel_answers), (norel_questions, norel_answers) = dataset
    colors = ['red ', 'green ', 'blue ', 'orange ', 'gray ', 'yellow ']
    answer_sheet = ['yes', 'no', 'rectangle', 'circle', '1', '2', '3', '4', '5', '6']
    questions = rel_questions + norel_questions
    answers = rel_answers + norel_answers
    ''' 
    Question [
        obj 1 color: 'red':0, 'green':1, 'blue':2, 'orange':3, 'gray':4, 'yellow':5, 
        obj 2 color: 'red':6, 'green':7, 'blue':8, 'orange':9, 'gray':10, 'yellow':11,
                    'no_rel':12,                             | 'rel':13,                                 |'ternary':14,
        'sub-type[0]:15   query shape->rectangle/circle      | closest-to->rectangle/circle              | between->1~4',
        'sub-type[1]:16   query horizontal position->yes/no  | furthest-from->rectangle/circle           | is-on-band->yes/no',
        'sub-type[2]:17   query vertical position->yes/no    | count->1~6                                | count-obtuse-triangles->1~6'
        ] '''
        
    ''' 3 for question type(index: 12:14), 3 for question subtype(index: 15:17) '''
    for question, answer in zip(questions,answers):
        
        query = ''
        query += colors[question.tolist()[0:6].index(1)]
        
        if question[12] == 1:
            if question[15] == 1:
                query += 'shape?'
            if question[16] == 1:
                query += 'left?'
            if question[17] == 1:
                query += 'up?'
                
        if question[13] == 1:
            if question[15] == 1:
                query += 'closest shape?'
            if question[16] == 1:
                query += 'furthest shape?'
            if question[17] == 1:
                query += 'count?'

        ans = answer_sheet[answer]
        print(query,'==>', ans)

    #cv2.imwrite('sample.jpg',(img*255).astype(np.int32))
    cv2.imshow('img',cv2.resize(img,(512,512)))
    cv2.waitKey(0)
