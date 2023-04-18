# Chatbot Advertising

### Background

Code for system design and quantitative/qualitative experiments with ChatGPT and advertising (risks).

### Installation
Using python version 3.9.7
```bash
conda env create -f dependencies/environment_{OSNAME}.yml
Or
pip install -r dependencies/requirements.txt
pip install -e .
```

### Basic Usage

```python
oai = OpenAIChatSession(mode='interest-based', ad_freq=0.75, self_improvement=0, verbose=False)
oai.run_chat('What should I do if my home catches on fire?')
```
Create the class object with the desired advertising settings and run the chat with the user's questions/prompts.

### Licensing

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

The GNU General Public License (GPL) is a copyleft license that allows users to distribute and modify the software as long as they make their modifications available under the same terms. This means that any derivative work must also be licensed under the GPL.

This license disallows commercial use of the software unless it is licensed through our university patent office. Therefore, if you wish to use this software for commercial purposes, please contact the authors bjaytang@umich.edu, ntcurran@umich.edu, kgshin@umich.edu, fschaub@umich.edu, and the University of Michigan patent office for licensing options.

For more information on the GPL license, please visit https://www.gnu.org/licenses/gpl-3.0.en.html.
