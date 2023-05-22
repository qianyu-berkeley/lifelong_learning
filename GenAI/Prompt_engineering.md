# ChatGPT Prompt Engineering for Developers

## Introduction

* Two types of LLMs
  * Base LLM
    * predict next words based on text training data
    * Does not always answer questions because training data may have a new question after a question not answers
  * Instruction Tuned LLM
    * Tried to following instructions
    * Fine-tune on instructions and good attemps at the instructions
    * RLHF: reinforcement learning with human feedback
    * Helpful, Honest, Harmless

## Guidelines

* Principle 1: be very clean and specific
  * clearn != short
  * Tactics:
    * Use delimiters: (e.g. `""", ```, ---, <>, <tag></tag>`)
      * to avoid prompt injections so we can seperate developer instruction from potential user instructions
    * Ask for structured output in prompt (e.g. HTML, JSON)
    * Check whether condistions are satisfied, check assumptions required to do the tasks (define assumption in the promopt)
    * Few-shot prompting: give successful example of completing tasks then ask model to perform the task
* Principle 2: Give the model time to "think"
  * Tactics:
    * Specific the steps required to complete a task
    * Instruct the model to work out its own solution before rushing to a conclusion
* Model limitations
  * Hallucination: make statements that sounds plausible but are not true
    * How to prevent?: first ask model to find relavant information then ane the question based on the relevant information
  
### Notes on using the OpenAI API outside of this classroom

To install the OpenAI Python library:

```bash
!pip install openai
```

The library needs to be configured with your account's secret key, which is available on the [website](https://platform.openai.com/account/api-keys). 

You can either set it as the `OPENAI_API_KEY` environment variable before using the library:

 ```
 !export OPENAI_API_KEY='sk-...'
 ```

Or, set `openai.api_key` to its value:

  ```python
  import openai
  openai.api_key = "sk-..."

  from dotenv import load_dotenv, find_dotenv
  _ = load_dotenv(find_dotenv()) # read local .env file

  openai.api_key  = os.getenv('OPENAI_API_KEY')

  # getting response by calling openai chat completion api
  def get_completion(prompt, model="gpt-3.5-turbo"):
      messages = [{"role": "user", "content": prompt}]
      response = openai.ChatCompletion.create(
          model=model,
          messages=messages,
          temperature=0, # this is the degree of randomness of the model's output
      )
      return response.choices[0].message["content"]
  ```

### A note about the backslash
- In the course, we are using a backslash `\` to make the text fit on the screen without inserting newline '\n' characters.
- GPT-3 isn't really affected whether you insert newline characters or not.  But when working with LLMs in general, you may consider whether newline characters in your prompt may affect the model's performance.  


## Iterative Prompt Development

* Prompt guidelines:
  * Be clear and specific
  * Analyze why result does not give desired output
  * Refine the idea and the prompt
    * Add constraints (word count, number of sentences etc)
    * Add more requirement (focus on certain aspect)
  * Repeat
* Iterative Process
  * Try something
  * Analyze where the result does not give what your want
  * Clarify instructions, give more time to think
  * Refine prompts with a batch of examples


## Summarizating

* Give the command as

  Example: 

  ```python
  prompt = f"""
  you task is to generate a short summary of ... 

  Summarize the review below, delimited by triple backticks, in at most x words, focus on y aspects that mentioned z
  
  Review: ```{prduct_review}```
  """
  ```

* Try `extract instead of `summarize`

  Example:

  ```python
  prompt = f"""
  You task is to extract relatvant information from ...

  From the review below, delimited by triple quotes extract the information relevant to x, limit to y words

  Review: ```{product_review}```
  """
  ```
  
* Can summerize multiple items, just use a loop

## Inferring

* tasks of analysis, extracting name (NER), sentiment analysis, extracting topics can be completed with a single model/api (speed)
* Infer sentiment 

  Example

  ```python
  prompt = f"""
  What is the sentiment of the following product review, 
  which is delimited with triple backticks?

  Give your answer as a single word, either "positive" \
  or "negative".
  
  Review text: '''{review}'''
  """
  ```

* Identify type of emotions

  ```python
  prompt = f"""
  Identify a list of emotions that the writer of the \
  following review is expressing. Include no more than \
  five items in the list. Format your answer as a list of \
  lower-case words separated by commas.

  Review text: '''{lamp_review}'''
  """
  ```

  ```python
  prompt = f"""
  Is the writer of the following review expressing anger?\
  The review is delimited with triple backticks. \
  Give your answer as either yes or no.

  Review text: '''{lamp_review}'''
  """
  ```

* Extract information (e.g. entities) from reviews and articles

  ```python
  prompt = f"""
  Identify the following items from the review text: 
  - Item purchased by reviewer
  - Company that made the item

  The review is delimited with triple backticks. \
  Format your response as a JSON object with
  "Item" and "Brand" as the keys. 
  If the information isn't present, use "unknown" \
  as the value.
  Make your response as short as possible.
    
  Review text: '''{lamp_review}'''
  """
  ```
  

* Perform multiple-tasks at once

  ```python
  prompt = f"""
  Identify the following items from the review text: 
  - Sentiment (positive or negative)
  - Is the reviewer expressing anger? (true or false)
  - Item purchased by reviewer
  - Company that made the item

  The review is delimited with triple backticks. \
  Format your response as a JSON object with \
  "Sentiment", "Anger", "Item" and "Brand" as the keys.
  If the information isn't present, use "unknown" \
  as the value.
  Make your response as short as possible.
  Format the Anger value as a boolean.

  Review text: '''{lamp_review}'''
  ```

* Inferring topics
  * Give a story in text or a http link, we can prompt to ask for extracting topics from the given source

  ```python
  story = """
  In a recent survey conducted by the government, 
  public sector employees were asked to rate their level 
  of satisfaction with the department they work at. 
  The results revealed that NASA was the most popular 
  department with a satisfaction rating of 95%.

  One NASA employee, John Smith, commented on the findings, 
  stating, "I'm not surprised that NASA came out on top. 
  It's a great place to work with amazing people and 
  incredible opportunities. I'm proud to be a part of 
  such an innovative organization."

  The results were also welcomed by NASA's management team, 
  with Director Tom Johnson stating, "We are thrilled to 
  hear that our employees are satisfied with their work at NASA. 
  We have a talented and dedicated team who work tirelessly 
  to achieve our goals, and it's fantastic to see that their 
  hard work is paying off."

  The survey also revealed that the 
  Social Security Administration had the lowest satisfaction 
  rating, with only 45% of employees indicating they were 
  satisfied with their job. The government has pledged to 
  address the concerns raised by employees in the survey and 
  work towards improving job satisfaction across all departments.
  """
  prompt = f"""
  Determine five topics that are being discussed in the \
  following text, which is delimited by triple backticks.

  Make each item one or two words long. 

  Format your response as a list of items separated by commas.

  Text sample: '''{story}'''
  """
  response = get_completion(prompt)
  print(response)
  ```
  
  ```python
  topic_list = [
    "nasa", "local government", "engineering", 
    "employee satisfaction", "federal government"
  ]
  prompt = f"""
  Determine whether each item in the following list of \
  topics is a topic in the text below, which
  is delimited with triple backticks.

  Give your answer as list with 0 or 1 for each topic.\

  List of topics: {", ".join(topic_list)}

  Text sample: '''{story}'''
  """
  response = get_completion(prompt)
  print(response)
  ```

## Transforming Tasks (e.g. Translation, spelling and grammar checking)

* Translation task


  ```python
  # Direct translation
  prompt = f"""
  Translate the following English text to Spanish: \ 
  ```Hi, I would like to order a blender```
  """

  # Detect language
  prompt = f"""
  Tell me which language this is: 
  ```Combien coûte le lampadaire?```
  """
  
  # multiple translation
  prompt = f"""
  Translate the following  text to French and Spanish
  and English pirate: \
  ```I want to order a basketball```
  """

  # Formal and informal
  prompt = f"""
  Translate the following text to Spanish in both the \
  formal and informal forms: 
  'Would you like to order a pillow?'
  """
  ```
  
* Universal translator

  ```python
  user_messages = [
    "La performance du système est plus lente que d'habitude.",  # System performance is slower than normal         
    "Mi monitor tiene píxeles que no se iluminan.",              # My monitor has pixels that are not lighting
    "Il mio mouse non funziona",                                 # My mouse is not working
    "Mój klawisz Ctrl jest zepsuty",                             # My keyboard has a broken control key
    "我的屏幕在闪烁"                                               # My screen is flashing
  ] 

  for issue in user_messages:
      prompt = f"Tell me what language this is: ```{issue}```"
      lang = get_completion(prompt)
      print(f"Original message ({lang}): {issue}")

      prompt = f"""
      Translate the following  text to English \
      and Korean: ```{issue}```
      """
      response = get_completion(prompt)
      print(response, "\n")
  ```

* Tone transformation

  ```python
  prompt = f"""
  Translate the following from slang to a business letter: 
  'Dude, This is Joe, check out this spec on this standing lamp.'
  """
  ```

* Format conversion (very useful)

  ```python
  data_json = { "resturant employees" :[ 
      {"name":"Shyam", "email":"shyamjaiswal@gmail.com"},
      {"name":"Bob", "email":"bob32@gmail.com"},
      {"name":"Jai", "email":"jai87@gmail.com"}
  ]}

  prompt = f"""
  Translate the following python dictionary from JSON to an HTML \
  table with column headers and title: {data_json}
  """
  ```

* Spell check and grammar checking

  ```python
  text = [ 
    "The girl with the black and white puppies have a ball.",  # The girl has a ball.
    "Yolanda has her notebook.", # ok
    "Its going to be a long day. Does the car need it’s oil changed?",  # Homonyms
    "Their goes my freedom. There going to bring they’re suitcases.",  # Homonyms
    "Your going to need you’re notebook.",  # Homonyms
    "That medicine effects my ability to sleep. Have you heard of the butterfly affect?", # Homonyms
    "This phrase is to cherck chatGPT for speling abilitty"  # spelling
  ]
  for t in text:
      prompt = f"""Proofread and correct the following text
      and rewrite the corrected version. If you don't find
      and errors, just say "No errors found". Don't use 
      any punctuation around the text:
      ```{t}```
      """
  ```

  ```python
  text = f"""
  Got this for my daughter for her birthday cuz she keeps taking \
  mine from my room.  Yes, adults also like pandas too.  She takes \
  it everywhere with her, and it's super soft and cute.  One of the \
  ears is a bit lower than the other, and I don't think that was \
  designed to be asymmetrical. It's a bit small for what I paid for it \
  though. I think there might be other options that are bigger for \
  the same price.  It arrived a day earlier than expected, so I got \
  to play with it myself before I gave it to my daughter.
  """
  prompt = f"proofread and correct this review: ```{text}```"
  response = get_completion(prompt)
  print(response)

  # display changes
  from redlines import Redlines

  diff = Redlines(text,response)
  display(Markdown(diff.output_markdown))
  ```
  
  ```python
  prompt = f"""
  proofread and correct this review. Make it more compelling. 
  Ensure it follows APA style guide and targets an advanced reader. 
  Output in markdown format.
  Text: ```{text}```
  """
  ```

## Expanding (take a short piece of text to generate a long text)

* Generate email

  ```python
  prompt = f"""
  You are a customer service AI assistant.
  Your task is to send an email reply to a valued customer.
  Given the customer email delimited by ```, \
  Generate a reply to thank the customer for their review.
  If the sentiment is positive or neutral, thank them for \
  their review.
  If the sentiment is negative, apologize and suggest that \
  they can reach out to customer service. 
  Make sure to use specific details from the review.
  Write in a concise and professional tone.
  Sign the email as `AI customer agent`.
  Customer review: ```{review}```
  Review sentiment: {sentiment}
  """
  response = get_completion(prompt)
  print(response)
  ```

* Set different temperatures

  * For task that require reliability and predictability use temperature = 0.0
  * For task that require creativity and variety use higher temperature but it will also introduce less exact response based on prompt

  ```python
  prompt = f"""
  You are a customer service AI assistant.
  Your task is to send an email reply to a valued customer.
  Given the customer email delimited by ```, \
  Generate a reply to thank the customer for their review.
  If the sentiment is positive or neutral, thank them for \
  their review.
  If the sentiment is negative, apologize and suggest that \
  they can reach out to customer service. 
  Make sure to use specific details from the review.
  Write in a concise and professional tone.
  Sign the email as `AI customer agent`.
  Customer review: ```{review}```
  Review sentiment: {sentiment}
  """
  response = get_completion(prompt, temperature=0.7)
  print(response)
  ```

## Chatbot

* How to setup a chatbot
  * define a couple of persona
    * system: sets behavior of assistant
    * assistant: chat model
    * user: you

```python
import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.getenv('OPENAI_API_KEY')

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
#     print(str(response.choices[0].message))
    return response.choices[0].message["content"]

messages =  [  
{'role':'system', 'content':'You are an assistant that speaks like Shakespeare.'},    
{'role':'user', 'content':'tell me a joke'},   
{'role':'assistant', 'content':'Why did the chicken cross the road'},   
{'role':'user', 'content':'I don\'t know'}  ]

messages =  [  
{'role':'system', 'content':'You are friendly chatbot.'},    
{'role':'user', 'content':'Hi, my name is Isa'}  ]

messages =  [  
{'role':'system', 'content':'You are friendly chatbot.'},    
{'role':'user', 'content':'Yes,  can you remind me, What is my name?'}  ]

messages =  [  
{'role':'system', 'content':'You are friendly chatbot.'},
{'role':'user', 'content':'Hi, my name is Isa'},
{'role':'assistant', 'content': "Hi Isa! It's nice to meet you. \
Is there anything I can help you with today?"},
{'role':'user', 'content':'Yes, you can remind me, What is my name?'}  ]
```

* Build a personal chatbot for pizza ordering

  ```python
  def collect_messages(_):
      prompt = inp.value_input
      inp.value = ''
      context.append({'role':'user', 'content':f"{prompt}"})
      response = get_completion_from_messages(context) 
      context.append({'role':'assistant', 'content':f"{response}"})
      panels.append(
          pn.Row('User:', pn.pane.Markdown(prompt, width=600)))
      panels.append(
          pn.Row('Assistant:', pn.pane.Markdown(response, width=600, style={'background-color': '#F6F6F6'})))
  
      return pn.Column(*panels)

  import panel as pn  # GUI
  pn.extension()

  panels = [] # collect display 

  context = [ {'role':'system', 'content':"""
  You are OrderBot, an automated service to collect orders for a pizza restaurant. \
  You first greet the customer, then collects the order, \
  and then asks if it's a pickup or delivery. \
  You wait to collect the entire order, then summarize it and check for a final \
  time if the customer wants to add anything else. \
  If it's a delivery, you ask for an address. \
  Finally you collect the payment.\
  Make sure to clarify all options, extras and sizes to uniquely \
  identify the item from the menu.\
  You respond in a short, very conversational friendly style. \
  The menu includes \
  pepperoni pizza  12.95, 10.00, 7.00 \
  cheese pizza   10.95, 9.25, 6.50 \
  eggplant pizza   11.95, 9.75, 6.75 \
  fries 4.50, 3.50 \
  greek salad 7.25 \
  Toppings: \
  extra cheese 2.00, \
  mushrooms 1.50 \
  sausage 3.00 \
  canadian bacon 3.50 \
  AI sauce 1.50 \
  peppers 1.00 \
  Drinks: \
  coke 3.00, 2.00, 1.00 \
  sprite 3.00, 2.00, 1.00 \
  bottled water 5.00 \
  """} ]  # accumulate messages


  inp = pn.widgets.TextInput(value="Hi", placeholder='Enter text here…')
  button_conversation = pn.widgets.Button(name="Chat!")

  interactive_conversation = pn.bind(collect_messages, button_conversation)

  dashboard = pn.Column(
      inp,
      pn.Row(button_conversation),
      pn.panel(interactive_conversation, loading_indicator=True, height=300),
  )

  dashboard


  messages =  context.copy()
  messages.append(
  {'role':'system', 'content':'create a json summary of the previous food order. Itemize the price for each item\
  The fields should be 1) pizza, include size 2) list of toppings 3) list of drinks, include size   4) list of sides include size  5)total price '},    
  )
  #The fields should be 1) pizza, price 2) list of toppings 3) list of drinks, include size include price  4) list of sides include size include price, 5)total price '},    

  response = get_completion_from_messages(messages, temperature=0)
  print(response)
  ```