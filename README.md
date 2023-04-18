# Chatbot Advertising

## Background

Code for system design and quantitative/qualitative experiments with ChatGPT and advertising (risks).

## Installation
Using python version 3.9.7
```bash
conda env create -f dependencies/environment_{OSNAME}.yml
```
Or
```bash
pip install -r dependencies/requirements.txt
pip install -e .
```

## Basic Usage

### Configuration

Configure your demographic data and initialize the chat session with the parameters you would like. `data/user_demographics.json` is the default file.

```json
{
    "age": 23,
    "gender": "Male",
    "relationship": "Single",
    "race": "Asian",
    "interests": ["Anime", "Video Games", "Academia", "Investing", "Mindfulness", "Software", "AI", "Research", "Manga"],
    "occupation": "Research Assistant",
    "politics": "Socially Left Classical Liberal",
    "religion": "Atheist with Buddhist influence",
    "location": "Ann Arbor, MI, USA"
}
```

### Running

```bash
./run.sh
```
Type "new_session" to create a new chat conversation. Type "load_session" to load an old chat conversation. Type "exit" to exit the chatbot.

### New chat session

Or type "new_session"

```python
oai.new_session()
```

### Populating products

```python
oai.populate_products()
```

## Licensing

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

The GNU General Public License (GPL) is a copyleft license that allows users to distribute and modify the software as long as they make their modifications available under the same terms. This means that any derivative work must also be licensed under the GPL.

This license disallows commercial use of the software unless it is licensed through our university patent office. Therefore, if you wish to use this software for commercial purposes, please contact the authors bjaytang@umich.edu, ntcurran@umich.edu, kgshin@umich.edu, fschaub@umich.edu, and the University of Michigan patent office for licensing options.

For more information on the GPL license, please visit https://www.gnu.org/licenses/gpl-3.0.en.html.
