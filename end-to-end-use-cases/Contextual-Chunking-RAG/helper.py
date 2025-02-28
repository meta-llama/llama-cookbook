import re
import io
import codecs
import random
from openai import OpenAI
from config import DEEPINFRA_API_KEY

openai = OpenAI(api_key=DEEPINFRA_API_KEY, base_url="https://api.deepinfra.com/v1/openai")
#client = OpenAI(api_key=OPENAI_API_KEY)


def file_put_contents(filename, st):
	file = codecs.open(filename, "w", "utf-8")
	file.write(st)
	file.close()

def file_get_contents(name):
	f = io.open(name, mode="r", encoding="utf-8") #utf-8 | Windows-1252
	return f.read()


def openai_run(system_prompt, user_message):
	messages = [{"role":"system", "content":system_prompt}, {"role":"user", "content":user_message}]    
	completion = client.chat.completions.create(
	  model="gpt-4o-mini", #"gpt-4o-2024-05-13",
	  temperature=0,
	  max_tokens=2000,
	  messages=messages
	)
	message = completion.choices[0].message
	return message.content    


def anthropic_run(system_prompt, user_message):
	import anthropic 
	client = anthropic.Anthropic(  
	api_key=ANTHROPIC_API_KEY,
	)
	message = client.messages.create(
	model="claude-3-sonnet-20240229", #"claude-3-opus-20240229",
	max_tokens=4096,
	system=system_prompt,
	messages=[
	 {"role": "user", "content": user_message}
	]
	)
	return message.content[0].text


def deepinfra_run(system_prompt, user_message):
	chat_completion = openai.chat.completions.create(
		model="meta-llama/Meta-Llama-3.1-405B-Instruct",
		messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
		max_tokens=4096
	)
	return chat_completion.choices[0].message.content



def get_llm_answer(chunks_content, user_message): #keywords + content
	gp = "Is answer is not given below, say that you don't know it. Make sure to copy answers from documents without changing them."+chunks_content
	answer = deepinfra_run(gp, user_message)
	return answer



def parse_keywords(content):
	result = []
	lines = content.strip().split('\n')
	current_chunk = None
	inline_pattern = re.compile(r'^\s*[^#:]+\s*:\s*(.+)$')  # Matches lines like "Chunk1: word1, word2"
	#section_pattern = re.compile(r'^###\s*[^#]+\s*###$') #v1
	section_pattern = re.compile(r'[#\*]*\s*Chunk\s*\d+\s*[#\*]*') #v2
 
	for line in lines:
		line = line.strip()
		if not line: continue
		inline_match = inline_pattern.match(line)

		if inline_pattern.match(line) and "Chunk" in line:			
			words_str = inline_match.group(1)
			words = [word.strip() for word in words_str.split(',') if word.strip()]
			result.append(words)

		elif section_pattern.match(line):			
			if current_chunk: result.append(current_chunk)
			current_chunk = []

		elif current_chunk is not None: #section_pattern continuation
			words = [word.strip() for word in line.split(',') if word.strip()]
			current_chunk.extend(words)

	if current_chunk: result.append(current_chunk)
	return result



def generate_contextual_keywords(chunked_content):
	system_prompt = '''
	Each chunk is separated as ### Chunk [id] ###. For each chunk generate keywords required to fully understand the chunk without any need for looking at the previous chunks.
	Don't just say "List of services", because its unclear what services are you referring to. Make sure to cover all chunks.
	Sample output:
	Chunk 1: BMW X5, pricings in France
	Chunk 2: BMW X5, discounts
	'''
	keywords_st = deepinfra_run(system_prompt, chunked_content)
	print("Keywords_st:\n", keywords_st, "\n")
	keywords = parse_keywords(keywords_st)    
	return keywords


def generate_questions_bychunk(chunks):
	system_prompt = '''
 Given a chunk from document. Generate 1-3 questions related to the chunk. Each question must be full and not require additional context. 
 Example output:
 1. How to open new account?
 2. How much BMW X5 costs? 
	'''	
	n = len(chunks)
	indexes = [i for i in range(n)]
	random.shuffle(indexes)
	for idx in indexes[: min(n//5, 60)]:
		chunk  = chunks[idx]
		text = "#"+(", ".join(chunk["keywords"]))+"\n"+chunk["content"]
		out =  deepinfra_run(system_prompt, text) #anthropic_run(system_prompt, text)
		question_pattern = re.compile(r'^\s*\d+\.\s+(.*)', re.MULTILINE)
		questions = question_pattern.findall(out)
		chunk["questions"] = questions
		chunk["idx"] = idx
	return chunks

	

def temp():
	st = '''
Here are the keywords for each chunk:

**Chunk 1**
3M, industrial and consumer products, electrical power transmission, renewable energy, infrastructure, Communication Markets Division, Germany

### Chunk 2 ###
3M, consumer retail, office supply products, home improvement products, Scotch brand, Post-it Products, Filtrete Filters, Thinsulate Insulation

** Chunk 3 **
3M, patents, trademarks, research and development, inventions, intellectual property, legal protection
'''
	print( parse_keywords(st) )



if __name__=="__main__":
	temp()

