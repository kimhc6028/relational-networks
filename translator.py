import cv2
def translate(dataset):
    img, (rel_questions, rel_answers), (norel_questions, norel_answers) = dataset
    colors = ['red ', 'green ', 'blue ', 'orange ', 'gray ', 'yellow ']
    answer_sheet = ['yes', 'no', 'rectangle', 'circle', '1', '2', '3', '4', '5', '6']
    questions = rel_questions + norel_questions
    answers = rel_answers + norel_answers

    print rel_questions
    print rel_answers


    for question,answer in zip(questions,answers):
        query = ''
        query += colors[question.tolist()[0:6].index(1)]

        if question[6] == 1:
            if question[8] == 1:
                query += 'shape?'
            if question[9] == 1:
                query += 'left?'
            if question[10] == 1:
                query += 'up?'
        if question[7] == 1:
            if question[8] == 1:
                query += 'closest shape?'
            if question[9] == 1:
                query += 'furthest shape?'
            if question[10] == 1:
                query += 'count?'

        ans = answer_sheet[answer]
        print query,'==>', ans
    #cv2.imwrite('sample.jpg',(img*255).astype(np.int32))
    cv2.imshow('img',cv2.resize(img,(512,512)))
    cv2.waitKey(0)
