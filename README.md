# LLM Chatbot Advertising

## Background

Code for system design and quantitative/qualitative experiments with ChatGPT and advertising (risks). For running our system just for demo purposes and ideation, you should use [https://chatbotumich.com/](https://chatbotumich.com/) with key `chatbotrtcl` (for disclosed ads) or `nosponsor` (for undisclosed ads). This repository is primarily just for deploying your own LLM advertising engine or reproducing evaluations.

## Installation

Using python version 3.9.7

```bash
pip install -r dependencies/requirements.txt
pip install -e .
```

Create a folder named `sensitive/`. In this folder, you should have an `objects.py` with a variable:
```python
openai_key = 'YOUR_OPENAI_API_KEY_HERE'
```

## Basic Usage

For the system design used in our user study and the prompting approaches, we provide details on running them below.

### Configuration

The chatbot takes as parameters: mode, ad_freq, model, self_improvement, verbose. For more information, run `python src/Chatbot.py --help`

### Running

```bash
./scripts/run.sh
```
Type "new_session" to create a new chat conversation. Type "load_session" to load an old chat conversation. Type "exit" to exit the chatbot.

### New Chat Session

Or type "new_session"

```python
oai.new_session()
```

### Populating Products

```python
oai.populate_products()
```

## Flask Web Server

```bash
cd website/
python website.py
```

For frontend website and next.js server, visit this repository: [https://github.com/byron123t/chatbot-ads-website](https://github.com/byron123t/chatbot-ads-website).

## Licensing

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

The GNU General Public License (GPL) is a copyleft license that allows users to distribute and modify the software as long as they make their modifications available under the same terms. This means that any derivative work must also be licensed under the GPL.

For more information on the GPL license, please visit https://www.gnu.org/licenses/gpl-3.0.en.html.
