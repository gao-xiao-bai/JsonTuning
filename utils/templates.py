
t0_question_answer = [
    ("{question}\n", "{answer}"),
    ("{question}\nAnswer:", "{answer}"),
    ("{question}\nA:", "{answer}"),
    ("Q:{question}\nA:", "{answer}"),
    ("Question: {question}\nAnswer:", "{answer}"),
    ("Answer the following question: {question}\nAnswer:", "{answer}"),
    ("Given the question: {question}\nThe answer is:", "{answer}"),
    ("{question}\nThe answer to this question is:", "{answer}"),
    ("Please answer the following question: {question}\nA:", "{answer}"),
    ("Please answer the following question: {question}\nAnswer:",
     "{answer}"),
]

t0_multiple_choice_separated_options = [
    ("{question}\n{options_}", "{answer}"),
    ("{question}\n{options_}\nAnswer:", "{answer}"),
    ("{question}\n\n{options_}\nAnswer:", "{answer}"),
    ("Q: {question}\n\n{options_}\nA:", "{answer}"),
    ("Answer the following question: {question}\n\n{options_}\nAnswer:",
     "{answer}"),
    ("{options_}\n\n{question}\nAnswer:", "{answer}"),
    ("{options_}\nQ: {question}\nA:", "{answer}"),
    ("{question}\n\n{options_}\nThe answer is:", "{answer}"),
    ("{options_}\nGiven those answer options, answer the "
     "question: {question}\nA:", "{answer}"),
    ("Q: {question}\n\n{options_}\nThe answer is:", "{answer}"),
]


NER_templates = [
    ("definition: {definition}\ntext: {text}\nentity categories: {entity categories}\nentities:", "{entities}"),
    ("definition: {definition}\nentity categories: {entity categories}\ntext: {text}\nentities:", "{entities}"),
    ("text: {text}\ndefinition: {definition}\nentity categories: {entity categories}\nentities:", "{entities}"),
    ("text: {text}\nentity categories: {entity categories}\ndefinition: {definition}\nentities:", "{entities}"),
    ("entity categories: {entity categories}\ntext: {text}\ndefinition: {definition}\nentities:", "{entities}"),
    ("entity categories: {entity categories}\ndefinition: {definition}\ntext: {text}\nentities:", "{entities}"),
    ("{definition}\ntext: {text}\nentity categories: {entity categories}\nentities:", "{entities}"),
    ("{definition}\nentity categories: {entity categories}\ntext: {text}\nentities:", "{entities}"),
    ("text: {text}\nentity categories: {entity categories}\n{definition}\nentities:", "{entities}"),
    ("entity categories: {entity categories}\ntext: {text}\n{definition}\nentities:", "{entities}"),
]


RE_templates_without_entity_category = [
    ("definition: {definition}\ntext: {text}\nrelations: {relations}\nrelational triplets:", "{relational triplets}"),
    ("definition: {definition}\nrelations: {relations}\ntext: {text}\nrelational triplets:", "{relational triplets}"),
    ("text: {text}\ndefinition: {definition}\nrelations: {relations}\nrelational triplets:", "{relational triplets}"),
    ("text: {text}\nrelations: {relations}\ndefinition: {definition}\nrelational triplets:", "{relational triplets}"),
    ("relations: {relations}\ntext: {text}\ndefinition: {definition}\nrelational triplets:", "{relational triplets}"),
    ("relations: {relations}\ndefinition: {definition}\ntext: {text}\nrelational triplets:", "{relational triplets}"),
    ("{definition}\ntext: {text}\nrelations: {relations}\nrelational triplets:", "{relational triplets}"),
    ("{definition}\nrelations: {relations}\ntext: {text}\nrelational triplets:", "{relational triplets}"),
    ("text: {text}\nrelations: {relations}\n{definition}\nrelational triplets:", "{relational triplets}"),
    ("relations: {relations}\ntext: {text}\n{definition}\nrelational triplets:", "{relational triplets}"),
]


RE_templates_with_entity_category = [
    ("definition: {definition}\ntext: {text}\nentity categories: {entity categories}\nrelations: {relations}\nrelational triplets:", "{relational triplets}"),
    ("definition: {definition}\nentity categories: {entity categories}\nrelations: {relations}\ntext: {text}\nrelational triplets:", "{relational triplets}"),
    ("definition: {definition}\ntext: {text}\nrelations: {relations}\nentity categories: {entity categories}\nrelational triplets:", "{relational triplets}"),
    ("definition: {definition}\nrelations: {relations}\nentity categories: {entity categories}\ntext: {text}\nrelational triplets:", "{relational triplets}"),
    ("text: {text}\ndefinition: {definition}\nentity categories: {entity categories}\nrelations: {relations}\nrelational triplets:", "{relational triplets}"),
    ("text: {text}\nentity categories: {entity categories}\nrelations: {relations}\ndefinition: {definition}\nrelational triplets:", "{relational triplets}"),
    ("text: {text}\ndefinition: {definition}\nrelations: {relations}\nentity categories: {entity categories}\nrelational triplets:", "{relational triplets}"),
    ("text: {text}\nrelations: {relations}\nentity categories: {entity categories}\ndefinition: {definition}\nrelational triplets:", "{relational triplets}"),
    ("entity categories: {entity categories}\nrelations: {relations}\ntext: {text}\ndefinition: {definition}\nrelational triplets:", "{relational triplets}"),
    ("relations: {relations}\nentity categories: {entity categories}\ndefinition: {definition}\ntext: {text}\nrelational triplets:", "{relational triplets}"),
]


EE_templates = [
    ("definition: {definition}\ntext: {text}\nevent categories: {event categories}\nargument categories: {argument categories}\nevents:", "{events}"),
    ("definition: {definition}\nevent categories: {event categories}\nargument categories: {argument categories}\ntext: {text}\nevents:", "{events}"),
    ("definition: {definition}\ntext: {text}\nargument categories: {argument categories}\nevent categories: {event categories}\nevents:", "{events}"),
    ("definition: {definition}\nargument categories: {argument categories}\nevent categories: {event categories}\ntext: {text}\nevents:", "{events}"),
    ("text: {text}\ndefinition: {definition}\nevent categories: {event categories}\nargument categories: {argument categories}\nevents:", "{events}"),
    ("text: {text}\nevent categories: {event categories}\nargument categories: {argument categories}\ndefinition: {definition}\nevents:", "{events}"),
    ("text: {text}\ndefinition: {definition}\nargument categories: {argument categories}\nevent categories: {event categories}\nevents:", "{events}"),
    ("text: {text}\nargument categories: {argument categories}\nevent categories: {event categories}\ndefinition: {definition}\nevents:", "{events}"),
    ("event categories: {event categories}\nargument categories: {argument categories}\ntext: {text}\ndefinition: {definition}\nevents:", "{events}"),
    ("argument categories: {argument categories}\nevent categories: {event categories}\ndefinition: {definition}\ntext: {text}\nevents:", "{events}"),
]


NL2SQL_templates = [
    ("definition: {definition}\nquestion: {question}\ndatabase schema: {database schema}\nSQL query:", "{SQL query}"),
    ("definition: {definition}\ndatabase schema: {database schema}\nquestion: {question}\nSQL query:", "{SQL query}"),
    ("question: {question}\ndefinition: {definition}\ndatabase schema: {database schema}\nSQL query:", "{SQL query}"),
    ("question: {question}\ndatabase schema: {database schema}\ndefinition: {definition}\nSQL query:", "{SQL query}"),
    ("database schema: {database schema}\nquestion: {question}\ndefinition: {definition}\nSQL query:", "{SQL query}"),
    ("database schema: {database schema}\ndefinition: {definition}\nquestion: {question}\nSQL query:", "{SQL query}"),
    ("{definition}\nquestion: {question}\ndatabase schema: {database schema}\nSQL query:", "{SQL query}"),
    ("{definition}\ndatabase schema: {database schema}\nquestion: {question}\nSQL query:", "{SQL query}"),
    ("question: {question}\ndatabase schema: {database schema}\n{definition}\nSQL query:", "{SQL query}"),
    ("database schema: {database schema}\nquestion: {question}\n{definition}\nSQL query:", "{SQL query}"),
]

