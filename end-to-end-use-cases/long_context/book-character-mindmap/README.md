# Book Character Mindmap

![Book Character Mindmap](public/mindmap.png)

Book Mind is a web application that allows users to explore character relationships and storylines in books using AI-powered visualizations.
This leverages Llama 4's impressive 10M and 1M token context windows to process entire books at once, enabling comprehensive analysis of complex narratives and character relationships across lengthy texts.

## Features

### Leverage Long Context Length
| Model | Meta Llama4 Maverick | Meta Llama4 Scout | OpenAI GPT-4.5 | Claude Sonnet 3.7 |
| ----- | -------------- | -------------- | -------------- | -------------- |
| Context Window | 1M tokens | 10M tokens | 128K tokens | 1K tokens | 200K tokens |

Because of the long context length, Book Mind can process entire books at once, providing a comprehensive understanding of complex narratives and character relationships.

- Interactive Mind Maps: Visualize relationships between characters and plot elements.
- Book Summaries: Get concise overviews of plots and themes.

## Getting Started

### Get API Key

Sign up for a free API key at [https://www.together.ai/](https://www.together.ai/).

Create a `.env` file in the root directory of the project and add the following line:

```
REACT_APP_TOGETHER_API_KEY=<YOUR_API_KEY>
```

### Frontend Setup

1. Install dependencies:

```
npm install
```

2. Run the application:

```
npm start
```

## Get Copyright Free Books

- [Project Gutenberg](https://www.gutenberg.org/)
  - [Romeo and Juliet](https://www.gutenberg.org/ebooks/1513): 50,687 input tokens
  - [Moby-Dick; The Whale](https://www.gutenberg.org/ebooks/2701): 318,027 input tokens
