import time
import torch
import pickle
import urllib.parse
#from env import BashEnv
#from NeuralAgent import NeuralAgent
#from play import play
import requests
import sys
import os
import csv
import argparse
from collections import Counter
from time import sleep

import gym
from selenium import webdriver
import chromedriver_binary
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options  
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
import configparser
from jinja2 import Environment, FileSystemLoader 
import numpy as np
import gym.spaces
import pexpect, getpass
import textworld
from textworld.core import GameState
#import re
import subprocess
import random
import xml.etree.ElementTree as ET
import re
from typing import List, Mapping, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

import textworld
import textworld.gym
from textworld import EnvInfos

#import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


#import os
from glob import glob
import matplotlib.pyplot as plt
#import gym
#import textworld.gym
import joblib
from html import escape
import base64
from bs4 import BeautifulSoup

command_list = []

options = Options()

config = configparser.ConfigParser()
config.read('config.ini')

window_width = int(config['Selenium']['window_width'])
window_height = int(config['Selenium']['window_height'])
position_width = int(config['Selenium']['position_width'])
position_height = int(config['Selenium']['position_height'])

html_dir = config['Common']['html_dir']
html_template = config['Common']['html_template']
html_file = config['Common']['ga_html_file']
print(html_file)
#options.add_argument('--headless') 
#nltk.download('punkt')

os.environ["http_proxy"] = "任意"




obj_browser = webdriver.Chrome(options=options) 
obj_browser.set_window_size(window_width, window_height)
obj_browser.set_window_position(position_width, position_height)




obj_browser.get('http://192.168.56.116/login.php')
id = obj_browser.find_element_by_name("username")
id.send_keys("admin")
password = obj_browser.find_element_by_name("password")
password.send_keys("password")

time.sleep(1)

# ログインボタンをクリック
login_button = obj_browser.find_element_by_name("Login")
login_button.click()



obj_browser.get('http://192.168.56.116/security.php')
#obj_browser.find_element_by_name("security").click()
x = obj_browser.find_element_by_name("security")
select = Select(x)
select.select_by_value('low')
obj_browser.find_element_by_name("seclev_submit").click()
env = Environment(loader=FileSystemLoader(html_dir))
template = env.get_template(html_template)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class BashEnv(textworld.Environment):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._process = None
        #self.prompt = r"##dkadfa09a2tafd##"
        self.prompt = ">"
        self.prompt2 = " "
        self.flag='Server username: root'
        self.ip_ = None
 
        self.datalist2 = None
        self.state = []
        self.allstate = []
        self.count = 0
        self.state_count = 0
        self.text = []
        self.episode_count = 0
    def close(self) -> None:
        if self.game_running:
            self._process.kill(9)
            self._process.wait()
            self._process = None

    def __del__(self):
        self.close()

    def load(self, ulx_file: str) -> None:
        self.close()  # Terminate existing process if needed.
        self._gamefile = ulx_file # 不要

        
    @property
    def game_running(self) -> bool:
        """ Determines if the game is still running. """
        return self._process is not None 
        
        
    def url_encode(self,token):
        s_quote = urllib.parse.quote(token)
        return s_quote
        
    def double_url_encode(self,token):
        s_quote = urllib.parse.quote(token)
        s_quote2 = urllib.parse.quote(s_quote)
        return s_quote2
        
    def unicode_encode(self,token):
        b = token.encode('unicode-escape')
        s_from_b_error = b.decode('utf-8')
        return s_from_b_error
        
    def html_encode(self, token):
        b = escape(token)
        return b
        
    def comment(self,token):
        b = token + '/**/'
        return b
        
    def junk(self,token):
        l = ['+-+-1-+-+', '!#$%&()*~+-_.,:;?@[/|\]^`']
        x = random.choice(l)
        b = token + x
        return b
        
    def base64(self,token):
        b = base64.b64encode(token.encode()).decode()
        return b
        
    def select_none(self,token):
        b = token
        return b
        
    def transform_character(self,token):
        list_token = list(token)
        token_length = len(list_token)
        token_length2 = token_length + 1
        index_token = list(range(token_length))
        list_length = list(range(1,token_length2))
        select_length = random.choice(list_length)
        select_index = random.sample(index_token, select_length)
        count = 0
        transform_token_list = []
        for x in index_token:
			    
            if count in select_index:
               token2 = list_token[count]
               token2 = str(token2)
               encode_token2 = token2.upper() 
               transform_token_list.append(encode_token2)
		       
            else:
                 token2 = list_token[count]
                 transform_token_list.append(token2)
			
            count += 1 
     #   print(transform_token_list)
        b = "".join(transform_token_list)
    #    print(b)
         
        return b
      # print(transform_token_list)
     #  result = "".join(transform_token_list)
   #    print(result)
        
    
    
    
    def null_select(self,token):
        list_token = list(token)
        token_length = len(list_token)
        token_length2 = token_length + 1
        index_token = list(range(token_length))
        list_length = list(range(1,token_length2))
        select_length = random.choice(list_length)
        select_index = random.sample(index_token, select_length)
        count = 0
        transform_token_list = []
        for x in index_token:
			    
            if count in select_index:
               token2 = list_token[count]
               token2 = str(token2)
               encode_token2 = token2 + '%00'
               transform_token_list.append(encode_token2)
		       
            else:
                 token2 = list_token[count]
                 transform_token_list.append(token2)
			
            count += 1 
     #   print(transform_token_list)
        b = "".join(transform_token_list)
   #     print(b)
         
        return b
    
    
    
    def space_select(self,token):
        list_token = list(token)
        token_length = len(list_token)
        token_length2 = token_length + 1
        index_token = list(range(token_length))
        list_length = list(range(1,token_length2))
        select_length = random.choice(list_length)
        select_index = random.sample(index_token, select_length)
        count = 0
        transform_token_list = []
        for x in index_token:
			    
            if count in select_index:
               token2 = list_token[count]
               token2 = str(token2)
               encode_token2 = token2 + ' '
               transform_token_list.append(encode_token2)
		       
            else:
                 token2 = list_token[count]
                 transform_token_list.append(token2)
			
            count += 1 
     #   print(transform_token_list)
        b = "".join(transform_token_list)
    #    print(b)
         
        return b
    
    def newline_select(self,token):
        list_token = list(token)
        token_length = len(list_token)
        token_length2 = token_length + 1
        index_token = list(range(token_length))
        list_length = list(range(1,token_length2))
        select_length = random.choice(list_length)
        select_index = random.sample(index_token, select_length)
        count = 0
        transform_token_list = []
        for x in index_token:
			    
            if count in select_index:
               token2 = list_token[count]
               token2 = str(token2)
               encode_token2 = token2 + '%0A'
               transform_token_list.append(encode_token2)
		       
            else:
                 token2 = list_token[count]
                 transform_token_list.append(token2)
			
            count += 1 
     #   print(transform_token_list)
        b = "".join(transform_token_list)
     #   print(b)
         
        return b
         
    def tab_select(self,token):
        list_token = list(token)
        token_length = len(list_token)
        token_length2 = token_length + 1
        index_token = list(range(token_length))
        list_length = list(range(1,token_length2))
        select_length = random.choice(list_length)
        select_index = random.sample(index_token, select_length)
        count = 0
        transform_token_list = []
        for x in index_token:
			    
            if count in select_index:
               token2 = list_token[count]
               token2 = str(token2)
               encode_token2 = token2 + '&Tab;'
               transform_token_list.append(encode_token2)
		       
            else:
                 token2 = list_token[count]
                 transform_token_list.append(token2)
			
            count += 1 
      #  print(transform_token_list)
        b = "".join(transform_token_list)
     #   print(b)
         
        return b
    
        
        
    def score(self,text):
       # payload = {'xss_text' : text}
        proxies = {
    'http' : '任意'
}

        r = requests.get("任意" + text, proxies=proxies)
      #  r = requests.get("任意" + text)
        print(r.status_code) 
      #  print(r.text)
        soup = BeautifulSoup(r.text)
        title=soup.find("title")
        try:
            x = title.text
            print(title.text)
        except:
              x = 'none'
              print('None')
       # print(title.text)
        if x == 'System error':
           reward = 0
           f = open('block5.txt', 'a', encoding='UTF-8') 
           f.write(str(text)+'\n')
           f.close()
	  # reward2 = np.array([[reward]], dtype=np.float32)
        
        
        elif r.status_code == 500:
             reward = 5
             f = open('500sucess5.txt', 'a', encoding='UTF-8') 
             f.write(str(text)+'\n')
             f.close()
             
        elif r.status_code == 502:
             reward = 5
             f = open('502sucess5.txt', 'a', encoding='UTF-8') 
             f.write(str(text)+'\n')
             f.close()
        else:
             reward = 5
             
	   #  reward2 = np.array([[reward]], dtype=np.float32)

        return reward
	
	
    def check_individual_selenium2(self, obj_browser, eval_html_path):
        # Evaluate running script using selenium.
        int_score = 0
        error_flag = False

        # Refresh browser for next evaluation.
        try:
            #obj_browser.refresh()
            obj_browser.get(eval_html_path)
            time.sleep(1)
            obj_browser.switch_to.alert.accept()
            obj_browser.refresh()
          #  obj_browser.refresh()
            int_score = 10
            print('alert')
          #  obj_browser.refresh()
        except Exception as e:
        
            print('no alert')
            int_score = 0


        return int_score#, error_flag
        
   
    def step(self, index,max_step,allstate):
       # print(allstate)
       # if not self.game_running:
           # raise GameNotRunningError()
        
        self.state_count += 1  
        state_index = self.state_count - 1
        
        token = allstate[state_index] 
      #  print(token)
        token = str(token)
        if token == 'SPACE':
           encoded_token = self.select_none(token)
           
        else:          
             if index == 0:
                encoded_token = self.url_encode(token)
		   
             elif index == 1:
                  encoded_token = self.double_url_encode(token)
		     
             elif index == 2:
                  encoded_token = self.unicode_encode(token)
		     
             elif index == 3:
                  encoded_token = self.html_encode(token)
		     
             elif index == 4:
                  encoded_token = self.comment(token)
		     
             elif index == 5:
                  encoded_token = self.junk(token)
		     
             elif index == 6:
                  encoded_token = self.base64(token)
		     
             elif index == 7:
                  encoded_token = self.select_none(token)
		     
             elif index == 8:
                  encoded_token = self.transform_character(token)
		     
             elif index == 9:
                  encoded_token = self.null_select(token)
		     
             elif index == 10:
                  encoded_token = self.space_select(token)
		     
             elif index == 11:
                  encoded_token = self.newline_select(token)
		     
             elif index == 12:
                  encoded_token = self.tab_select(token)
		     
      #  print(encoded_token)
        self.state = GameState()
     #   print(self.state)
        self.state.score = 0
        self.state.raw = encoded_token
        self.text.append(encoded_token)
      #  print(self.state.raw)
        if max_step == self.state_count:
           Y3 = ''.join(self.text)
           Y3 = Y3.replace('SPACE', ' ')
           print(Y3)
           reward1 = self.score(Y3)
           eval_html_path = "http://192.168.56.116/vulnerabilities/xss_r/?name=" + Y3
          # eval_html_path = '任意' + Y3
           reward2 = self.check_individual_selenium2(obj_browser,eval_html_path)
           self.state.score = reward1 + reward2
           print(self.state.score)
           if self.state.score == 5:
              f = open('antiwafhonban5.txt', 'a', encoding='UTF-8') 
              f.write(str(Y3)+'\n')
              f.close()
           
                
           elif self.state.score == 15:
                f = open('antiwafsucesshonban5.txt', 'a', encoding='UTF-8') 
                f.write(str(Y3)+'\n')
                f.close()
                
        if self.state.raw is None:
            raise GameNotRunningError()
      #  self.state.score = 30 if self.won() else -10 if self.lost() else -1
        
 
        self.state.done = False
     #   print(self.state.done)
      #  self.state.feedback = _strip_input_prompt_symbol(self.state.raw)
        if self.state_count == max_step:
           self.state_count = 0
           self.text = []
           
        return self.state.raw, self.state.score, self.state.done

    
    
            
    
            
            
              


    def reset(self) -> str:
        self.close()  # Terminate existing process if needed.
        self.allstate = None
        print('resetされました')
        
        self.episode_count += 1
        print(self.episode_count)
        self.count += 1
        count = 0
        with open('inputwakatiwaf2.txt') as f:
             for line in f:
                 count += 1
                 if self.count == count:
                   # x = []
                    line = line.replace('\n', '')
                    self.allstate = line.split(' ')
             
             if self.count == count:
                self.count = 0
                
              
        
        max_step = len(self.allstate)         
        if self.count == 910:
           self.count = 0
        

        print(self.allstate)
        obs = str(self.allstate[0])#self._process.before.decode('utf-8')
 
        return  obs, max_step ,self.allstate

    def render(self, mode: str = "human") -> None:
        outfile = StringIO() if mode in ['ansi', "text"] else sys.stdout

        msg = self.state.feedback.rstrip() + "\n"
        if self.display_command_during_render and self.state.last_command is not None:
            msg = '> ' + self.state.last_command + "\n" + msg

        # Wrap each paragraph.
        if mode == "human":
            paragraphs = msg.split("\n")
            paragraphs = ["\n".join(textwrap.wrap(paragraph, width=80)) for paragraph in paragraphs]
            msg = "\n".join(paragraphs)

        outfile.write(msg + "\n")

        if mode == "text":
            outfile.seek(0)
            return outfile.read()

        if mode == 'ansi':
            return outfile
    
 
    
 
        
#device = 'cpu'
        
        
class CommandScorer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CommandScorer, self).__init__()
        torch.manual_seed(42)  # For reproducibility
        self.embedding    = nn.Embedding(input_size, hidden_size)
     #   self.encoder_gru  = nn.GRU(hidden_size, hidden_size)
     #   self.cmd_encoder_gru  = nn.GRU(hidden_size, hidden_size)
        self.state_gru    = nn.GRU(hidden_size, hidden_size)
        self.hidden_size  = hidden_size
        self.state_hidden = torch.zeros(1, 1, hidden_size, device=device)
        self.critic       = nn.Linear(hidden_size, 1)
      #  self.att_cmd      = nn.Linear(hidden_size * 2, 1)
        self.action_head = nn.Linear(hidden_size, 13)
    def forward(self, obs, **kwargs):
        input_length = obs.size(0)
        batch_size = obs.size(1)
      #  print(obs)
       # nb_cmds = commands.size(1)

        embedded = self.embedding(obs)
      #  print(embedded.size())
     #   encoder_output, encoder_hidden = self.encoder_gru(embedded)
        state_output, state_hidden = self.state_gru(embedded, self.state_hidden)
     #   print(state_output.size())
        self.state_hidden = state_hidden
      #  state_hidden2 = torch.reshape(state_hidden, (1,1,256))
    #    state_output2 = torch.reshape(state_hidden, (1,1,256))
        value = self.critic(state_output)
     #   print(value.size())
     #   print(value)

        # Attention network over the commands.
      #  cmds_embedding = self.embedding.forward(commands)
     #   _, cmds_encoding_last_states = self.cmd_encoder_gru.forward(cmds_embedding)  # 1 x cmds x hidden

        # Same observed state for all commands.
     #   cmd_selector_input = torch.stack([state_hidden] * nb_cmds, 2)  # 1 x batch x cmds x hidden

        # Same command choices for the whole batch.
     #   cmds_encoding_last_states = torch.stack([cmds_encoding_last_states] * batch_size, 1)  # 1 x batch x cmds x hidden

        # Concatenate the observed state and command encodings.
      #  cmd_selector_input = torch.cat([cmd_selector_input, cmds_encoding_last_states], dim=-1)

        # Compute one score per command.
     #   scores = F.relu(self.att_cmd(cmd_selector_input)).squeeze(-1)  # 1 x Batch x cmds
        scores = self.action_head(state_hidden)
        probs = F.softmax(scores, dim=2)#F.softmax(scores, dim=2)  # 1 x Batch x cmds
      #  print(probs)
        index = probs[0].multinomial(num_samples=1).unsqueeze(0) # 1 x batch x indx
      #  print(index)
        return scores, index, value

    def reset_hidden(self, batch_size):
        self.state_hidden = torch.zeros(1, batch_size, self.hidden_size, device=device)


class NeuralAgent:
    """ Simple Neural Agent for playing TextWorld games. """
    MAX_VOCAB_SIZE = 10000
    UPDATE_FREQUENCY = 30
    LOG_FREQUENCY = 100
    GAMMA = 0.9
    
    def __init__(self) -> None:
        self._initialized = False
        self._epsiode_has_started = False
        self.id2word = ["<PAD>", "<UNK>"]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}
        
        self.model = CommandScorer(input_size=self.MAX_VOCAB_SIZE, hidden_size=128)
        self.optimizer = optim.Adam(self.model.parameters(), 0.0005)
        
        self.mode = "train"
        self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
        self.transitions = []
        self.model.reset_hidden(1)
        self.last_score = 0
        self.no_train_step = 0
    
    def train(self):
        self.mode = "train"
        self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
        self.transitions = []
        self.model.reset_hidden(1)
        self.last_score = 0
        self.no_train_step = 0
    
    def test(self):
        self.mode = "test"
        self.model.reset_hidden(1)
        
    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(description=True, inventory=True,objective=True,entities=True,location=True, admissible_commands=True,
                        won=True, lost=True)
    
    def _get_word_id(self, word):
        if word not in self.word2id:
            if len(self.word2id) >= self.MAX_VOCAB_SIZE:
                return self.word2id["<UNK>"]
            
            self.id2word.append(word)
            self.word2id[word] = len(self.word2id)
            
        return self.word2id[word]
            
    def _tokenize(self, text):
        # Simple tokenizer: strip out all non-alphabetic characters.
     #   text = re.sub("[^a-zA-Z0-9\- ]", " ", text)
   
       # text = ''.join(text)
        text = text.replace(' ', '')
        word_ids = list(map(self._get_word_id, text.split()))
        return word_ids

    def _process(self, texts):
        texts = list(map(self._tokenize, texts))
        max_len = max(len(l) for l in texts)
        padded = np.ones((len(texts), max_len)) * self.word2id["<PAD>"]

        for i, text in enumerate(texts):
            padded[i, :len(text)] = text

        padded_tensor = torch.from_numpy(padded).type(torch.long).to(device)
        padded_tensor = padded_tensor.permute(1, 0) # Batch x Seq => Seq x Batch
        return padded_tensor
      
    def _discount_rewards(self, last_values):
        returns, advantages = [], []
        R = last_values.data
        for t in reversed(range(len(self.transitions))):
            rewards, _, _, values = self.transitions[t]
            R = rewards + self.GAMMA * R
          #  print(R.size())
        #    print(values.size())
            adv = R - values
            returns.append(R)
            advantages.append(adv)
            
        return returns[::-1], advantages[::-1]

    def act(self, obs: str, score: int, done: bool) -> Optional[str]:
        global policy
        global value
        global entropy
        
        policy,value,entropy = [], [], []
        
        # Build agent's observation: feedback + look + inventory.
        input_ = obs
      #  print(obs)
        # Tokenize and pad the input and the commands to chose from.
        input_tensor = self._process([input_])
     #   commands_tensor = self._process(infos["admissible_commands"])
        
        # Get our next action and value prediction.
        outputs, indexes, values = self.model(input_tensor)
      #  action = infos["admissible_commands"][indexes[0]]
        action = indexes[0]
        if self.mode == "test":
            if done:
                self.model.reset_hidden(1)
            return action
        
        self.no_train_step += 1
      #  print(self.no_train_step)
        
        if self.transitions:
            reward = score - self.last_score  # Reward is the gain/loss in score.
          #  print(reward)
            self.last_score = score
 
 
            self.transitions[-1][0] = reward  # Update reward information.
          #  print(reward)
        
        self.stats["max"]["score"].append(score)
        if self.no_train_step % self.UPDATE_FREQUENCY == 0:
            # Update model
            returns, advantages = self._discount_rewards(values)
            
            loss = 0
            for transition, ret, advantage in zip(self.transitions, returns, advantages):
                reward, indexes_, outputs_, values_ = transition
               
              #  with open('reward.txt', 'a') as f:
                   # f.write(str(reward)+'\n')
              #  f.close()
                
                advantage        = advantage.detach() # Block gradients flow here.
                probs            = F.softmax(outputs_, dim=2)
                log_probs        = torch.log(probs)
                log_action_probs = log_probs.gather(2, indexes_)
                policy_loss      = (-log_action_probs * advantage).sum()
                value_loss       = (.5 * (values_ - ret) ** 2.).sum()
                entropy     = (-probs * log_probs).sum()
                loss += policy_loss + 0.5 * value_loss - 0.1 * entropy
                
                self.stats["mean"]["reward"].append(reward)
               
              #  print(self.stats["mean"]["reward"])
                self.stats["mean"]["policy"].append(policy_loss.item())
                #policy.append(self.stats["mean"]["policy"])
                self.stats["mean"]["value"].append(value_loss.item())
                #value.append(value_loss.item())
                self.stats["mean"]["entropy"].append(entropy.item())
                #entropy.append(entropy.item())
                self.stats["mean"]["confidence"].append(torch.exp(log_action_probs).item())
                #print(self.stats["mean"])
                #print(policy)
                
                
        
            
            if self.no_train_step % self.LOG_FREQUENCY == 0:
                msg = "{}. ".format(self.no_train_step)
                msg += "  ".join("{}: {:.3f}".format(k, np.mean(v)) for k, v in self.stats["mean"].items())
                msg += "  " + "  ".join("{}: {}".format(k, np.max(v)) for k, v in self.stats["max"].items())
                msg += "  vocab: {}".format(len(self.id2word))
                print("\n"+msg)
#                print(probs)
#                print(log_probs)
#                print(log_action_probs)
                self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 40)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
            self.transitions = []
            self.model.reset_hidden(1)
        else:
            # Keep information about transitions for Truncated Backpropagation Through Time.
            self.transitions.append([None, indexes, outputs, values])  # Reward will be set on the next call
        
        if done:
            self.last_score = 0  # Will be starting a new episode. Reset the last score.
            self.model.reset_hidden(1) # 追加
        
        return action
 




    
    
    
    
    
    
    

def play(agent, env, pathNo, max_step=300, nb_episodes=1, verbose=True):

    infos_to_request = agent.infos_to_request
    infos_to_request.max_score = True  # Needed to normalize the scores.
    
   
    # Collect some statistics: nb_steps, final reward.
    avg_moves, avg_scores, avg_norm_scores, avg_stepreward = [], [], [], []
 #   obs, infos = env.reset()  # Start new episode.
 #   env.set_flag(pathNo)
#    env.set_pwd()
#    print('Flag: '+env.flag)
  #  print("Current: "+ env.pwd())
    
    
    episodes = 0
    for no_episode in range(nb_episodes):
        episodes += 1
        print(episodes)
        obs,max_step,allstate = env.reset()  # Start new episode.\
        #env.scan(self)
       # env.set_flag(pathNo)
       # commandx = []
        
        score = 0
        done = False
        nb_moves = 0
        while not done:
            action_index = agent.act(obs, score, done)
 
            obs, score, done = env.step(action_index,max_step,allstate)

            
            
            nb_moves += 1
 
            if nb_moves >= max_step:
               done = True
        
        agent.act(obs, score, done)  # Let the agent know the game is done.
            
                
        if verbose:
            #command_list.clear()
            if score >= 100:
            
                print(" {}".format(nb_moves), end="")
            else:
                print(" {}".format(nb_moves)+".", end="")
        reward_step = score/nb_moves

        avg_stepreward.append(reward_step)
       # nb_moves2 = str(nb_moves) + 'n'
        avg_moves.append(nb_moves)
        
        
    #    print(avg_moves)
      #  avg_scores.append(score)
   #     print(avg_scores)
      #  avg_norm_scores.append(score / infos["max_score"])


   # joblib.dump(agent,'trained_agentX.pkl', compress=True)        

    #env.close()
  #  msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
  #  print(msg.format(np.mean(avg_moves), np.mean(avg_scores), infos["max_score"]))
    
    
    
    
    #return avg_moves, avg_scores

        
    
    
    
    

        
        
        
        
        
        
        
        
        
        
        
        
        









torch.cuda.is_available()

torch.set_default_tensor_type('torch.cuda.FloatTensor')
print(torch.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#env = BashEnv()
#for i in range(470):
env = BashEnv()
    #env = BashEnv()
#try:
  #  agent = joblib.load('trained_agentX.pkl')
#except:
    
    
agent = NeuralAgent()


#model=NeuralAgent()
starttime = time.time()
play(agent, env, -5, max_step=300, nb_episodes=150000, verbose=True).to(device)
#torch.save(agent.state_dict(), '/home/yoneda/metasploitenv/model3.pkl')
