import json
import string

genome_data_file = '/datasets01/VisualGenome1.2/061517/question_answers.json'
genome_questions_file = '/private/home/nvivek/VQA/orig_data/v2_OpenEnded_mscoco_genome_questions.json'
genome_annotations_file = '/private/home/nvivek/VQA/orig_data/v2_mscoco_genome_annotations.json'

translator=str.maketrans('','',string.punctuation)
with open(genome_data_file, 'r') as f:
    genome_data = json.load(f)

genome_questions = []
genome_annotations = []

# question keys
# image_id
# question
# question_id

# annotation keys
# question_id
# image_id
# multiple_choice_answer
# answers = []
    # answer
    # answer_id = 1:10
    # answer_confidence

for data in genome_data:
    all_qas = data['qas']
    for qas in all_qas:
        question = {}
        annotation = {}
        question['image_id'] = qas['image_id']
        # assume unique question_id for every question answer pair
        question['question_id'] = qas['qa_id']
        question['question'] = qas['question']
        genome_questions.append(question)
        annotation['image_id'] = qas['image_id']
        annotation['question_id'] = qas['qa_id']
        answertxt = qas['answer'].translate(translator)
        answertxt = answertxt.lower()
        annotation['multiple_choice_answer'] = answertxt
        annotation['answers'] = []
        for i in range(10):
            answer = {}
            answer['answer'] = answertxt
            answer['answer_confifence'] = 'yes'
            answer['answer_id'] = i + 1
            annotation['answers'].append(answer)
        genome_annotations.append(annotation)

genome_data = {}
genome_data['questions'] = genome_questions

with open(genome_questions_file, 'w') as f:
    json.dump(genome_data, f)

genome_data = {}
genome_data['annotations'] = genome_annotations

with open(genome_annotations_file, 'w') as f:
    json.dump(genome_data, f)
