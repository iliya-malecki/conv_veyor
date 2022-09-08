# %%
import pandas as pd
import numpy as np
import os
import re
import wikipedia
import unicodedata as ud
import PIL
from PIL import ImageFont, ImageDraw, Image
import cv2
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor

fonts_dir = './fonts/'
wikipedia.set_lang('uk')
ukr_syms = list('йцукенгшщзхїфівапролджєячсмитьбює0987654321-=+*/----___.,.,,.        ')

# %%
def get_bw(image):
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    return black, white


def dots_noise(image, prob, scale=5):

    output = image.copy() 
    black, white = get_bw(image)

    probs = np.random.random((np.array(output.shape[:2])/scale).astype(int))
    probs = cv2.resize(probs,(output.shape[1],output.shape[0]),cv2.INTER_AREA)

    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white

    return output


def sp_noise(image, prob, colors='rgbkw'):
    try:
        if len(prob) != len(colors):
            raise ValueError('When providing all probs explicitly, one probability per color is expected')
        splits = np.cumsum(np.r_[0, prob])
    except TypeError:
        splits = np.linspace(0, prob, len(colors)+1)

    black, white = get_bw(image)
    color_values = {
        'r': np.array([255,0,0]),
        'g': np.array([0,255,0]),
        'b': np.array([0,0,255]),
        'k': black,
        'w': white,
    }

    output = image.copy() 
    probs = np.random.random((np.array(output.shape[:2])).astype(int))

    for smol, big, c in zip(splits, splits[1:], colors):
        output[(smol < probs) & (probs <= big)] = color_values[c]

    return output

# %%
def obfuscate_image(image):
    for f, kwargs in [
        # (np.asarray, {}),
        # (dots_noise, {'prob':0.2}),
        # (sp_noise, {'prob':0.1}),
        # (Image.fromarray, {})
    ]:
        image = f(image, **kwargs)
    return image


# %%
def obfuscate_image2(image):
    for f, args in [
        (np.asarray, []),
        (cv2.cvtColor, [cv2.COLOR_RGB2GRAY]),
        (sp_noise, {'prob':0.05, 'colors':'k'}),
        (cv2.blur,[(3,3),5]),
        (lambda x: cv2.threshold(x,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1], ()),
        (cv2.blur,[(2,2),10]),
        (cv2.filter2D, {'ddepth':-1,'kernel':np.array([[0,-1,0],[-1,6,-1],[0,-1,0]])}),
        (Image.fromarray, {}),
        (lambda x: x.resize((np.array(x.size) / 1.2).astype(int)),()),
    ]:
        if isinstance(args, dict):
            image = f(image, **args)
        else:
            image = f(image, *args)
    return image

def obfuscate_image3(image):
    for f, args in [
        (lambda x: x.resize((np.array(x.size) / 2).astype(int)),()),
        (lambda x: x.resize((np.array(x.size) * 2).astype(int)),()),
        (np.asarray, []),
        (cv2.cvtColor, [cv2.COLOR_RGB2GRAY]),
        (lambda x: cv2.threshold(x,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)[1], ()),
        (lambda x: cv2.addWeighted(x, 1.5, cv2.GaussianBlur(x,(3,3),5),-0.5, 0), ()),
        (Image.fromarray, {}),
    ]:
        if isinstance(args, dict):
            image = f(image, **args)
        else:
            image = f(image, *args)
    return image


# %%
class Text2ImgGen:
    def __init__(self, text, fonts, w=(100,320), h=(48,56), size=46, max_char=25, colorgen=lambda:('black', 'white')) -> None:
        self.text = text
        self.fonts = [ImageFont.truetype(fonts + f, size) for f in os.listdir(fonts)]
        self.w = w
        self.h = h
        self.size = size
        self.max_char = max_char
        self.colorgen = colorgen

    def __iter__(self):
        return self

    def __next__(self):

        if len(self.text) < 1:
            raise StopIteration

        font = np.random.choice(self.fonts)

        foreground, background = self.colorgen()

        w = np.random.randint(*self.w)
        h = np.random.randint(*self.h)
        w_cut = 10
        h_cut = 50
        for n in range(1, self.max_char):
            if font.getlength(self.text[:n]) > w - self.size/2:
                break
        
        text, self.text = self.text[:n], re.sub(r'.*? ','',self.text[n:], count=1)
        
        if np.random.ranf() < 0.3:
            top = ''
        else:
            top = ''.join(np.random.choice(ukr_syms, len(text)+1))
        
        if np.random.ranf() < 0.3:
            bot = ''
        else:
            bot = ''.join(np.random.choice(ukr_syms, len(text)+1))

        display_text = top + '\n' + text + '\n' + bot
        
        im = Image.new("RGB",(w+w_cut*2,h+h_cut*2), background)
        draw = ImageDraw.Draw(im)

        draw.multiline_text(
            ((w+w_cut)/2,(h+h_cut)/2), 
            display_text, 
            fill=foreground, 
            font=font, 
            align='center', 
            anchor='mm', 
            spacing=np.random.randint(-1,5)
        )

        if not top and np.random.ranf() < 0.5:
            y = np.random.randint(h_cut/2+1, h_cut/2+8)
            draw.line((0, y, w+w_cut, y), fill=foreground, width=np.random.randint(1, 3))

        if not top and np.random.ranf() < 0.5:
            y = np.random.randint(h+h_cut/2-8, h+h_cut/2-1)
            draw.line((0, y, w+w_cut, y), fill=foreground, width=np.random.randint(1, 3))
            
        possible_angle_dist_std = np.rad2deg(np.arctan((h-self.size)/w*1.5))/3
        im = (
            im
            .rotate(
                np.random.normal(0, possible_angle_dist_std), 
                PIL.Image.Resampling.BICUBIC,
                fillcolor=background
            )
            .crop((w_cut/2, h_cut/2, w+w_cut/2, h+h_cut/2))
        )
        
        return obfuscate_image(im), text, ''.join([el[:2] for el in (' '.join(font.getname()).split())])


# %%
def generate_random_text(char_dict_path, word_count:'int', word_len:'tuple[int,int]'):
    with open(char_dict_path) as f:
        chars = f.read()
    chars = list(chars.replace('\n',''))
    
    return ''.join(
        np.concatenate([
            np.r_[
                np.random.choice(chars, size=np.random.randint(*word_len)),
                np.array([' '])
            ]
            for n in range(word_count)
        ])
    )
#%%
# %%
def generate_dataset(ttsplit, text, starting_n):

    with ThreadPoolExecutor(100) as exer:
        
        for img, label, fontname in Text2ImgGen(
            text,
            fonts_dir,
            colorgen=lambda:((10,10,10), (250,250,250))
        ):
            name = f'Word_{starting_n}_{fontname}.png'
            coinflip = ttsplit['train' if np.random.ranf() > 0.1 else 'test']
            
            exer.submit(lambda: img.save(coinflip['data']/name))
            print(
                f"{coinflip['data'].stem}/{name}\t{label}",
                file=coinflip['labels']
            )
            starting_n+=1
        exer.shutdown(wait=True)

    return starting_n
# %%

def main_generating_sequence(root):
    topics = [
        'запоріжсталь',
        'каметсталь',
        'інвойс',
        'податки',
        'ідентифікатор',
        'ГОК',
        'клієнт',
        'таблиця',
        'коди',
        'штрих-код',
        'QR-код',
        'рахунок',
        'фактура'
    ]

    allowed = pd.Series(list('ёйїЁЙЇ')).str.normalize('NFC').values

    root = Path(root).resolve()
    root.mkdir(exist_ok=True)
    train_data_dir = root/'train'
    train_data_dir.mkdir(exist_ok=True)
    test_data_dir = root/'test'
    test_data_dir.mkdir(exist_ok=True)
    

    with open(root/'rec_uk_train.txt', 'a') as train_labels, open(root/'rec_uk_test.txt', 'a') as test_labels:

        ttsplit_files = {
            'train':{'data':train_data_dir,'labels':train_labels},
            'test': {'data':test_data_dir,'labels':test_labels},
        }

        written_file_count = generate_dataset(
            ttsplit=ttsplit_files, 
            text=generate_random_text(
                './PaddleOCR/ppocr/utils/dict/uk_dict_ours.txt',
                100,
                (3,8)
            ),
            starting_n=0
            )
    raise
    if False:
        for topic in topics:
            for idea in np.random.permutation(wikipedia.search(topic, results=10)):

                try:
                    page = wikipedia.page(idea)
                except wikipedia.DisambiguationError as e:
                    try:
                        page = wikipedia.page(np.random.choice(e.options))
                    except Exception as e:
                        print('skipping a page because of an unrecoverable wikipedia error:')
                        print(e)
                        continue
                    
                content = pd.Series(list(ud.normalize('NFKD', re.sub('\n','', page.content))))
                questionable = content[content.apply(ud.combining)>0]
                bad_index = questionable[
                    ~
                    (content.shift(1)[questionable.index] + questionable)
                    .str.normalize('NFC').isin(allowed)
                ].index
                
                written_file_count = generate_dataset(
                    ttsplit=ttsplit_files,
                    text=ud.normalize('NFC',''.join(content.drop(index=bad_index))),
                    starting_n=written_file_count+1
                )

    print('done')
    os.system(f'ls {train_data_dir} > {root}/train_index.txt')
    os.system(f'ls {test_data_dir} > {root}/test_index.txt')



# %%
def heal_absent_images(root, data, labels):
    root = Path(root)
    datafiles = pd.Index(
        pd.concat((
            pd.read_csv(root/n).iloc[:,0]
            for n in data
        )).values
    )
    for labelfile in labels:
        with open(root/labelfile, 'r') as f:
            labeltext = f.read()
        labelset = (
            pd.read_csv(root/labelfile, sep='\t',header=None).iloc[:,0]
            .str.split('/',expand=True).iloc[:,-1]
        )
        bad = pd.Index(labelset.values).difference(pd.Index(datafiles))
        print(bad)
        for name in bad:
            labeltext = re.sub(fr'\n.*{re.escape(name)}.*\n','\n', labeltext)
        with open(root/labelfile, 'w') as f:
            f.write(labeltext)



# %%
def split_dataset_labels(n, source, leftover, new=None):
    with open(source) as f:
        lines = np.array(f.read().split('\n'))

    if n >= len(lines):
        raise ValueError('n >= len(lines)')
    
    chosen = np.random.randint(0, len(lines), size=n)
    split1, split2 = lines[chosen], np.delete(lines, chosen)
    with open(leftover, 'w') as f:
        print('\n'.join(split2), file=f)
    with open(new if new is not None else source, 'w') as f:
        print('\n'.join(split1), file=f)




root = './PaddleOCR/train_data/generated_images'
root = './test_images'

main_generating_sequence(root)

heal_absent_images(
    root,
    ['train_index.txt','test_index.txt'],
    ['rec_uk_train.txt','rec_uk_test.txt']
)

split_dataset_labels(
    1000, 
    f'{root}/rec_uk_test.txt', 
    f'{root}/rec_uk_test_backup.txt'
)
