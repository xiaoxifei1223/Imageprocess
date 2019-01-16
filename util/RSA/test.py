#_*_coding:utf-8_*_
from Crypto.Cipher import AES
from Crypto import Random
from time import time
import six
# import cv2
key = Random.new().read(AES.block_size) # initial key
iv = Random.new().read(AES.block_size)  # vector for key every turn

with open(r"G:\method_test_save\AES\307-20180614137.png", 'rb') as r:
    input_data = r.read()
r.close()
# input_data = input_file.read()
# input_file.close()

cfb_cipher = AES.new(key, AES.MODE_CFB, iv) # generate a encryptor
starttime = time()
enc_data = cfb_cipher.encrypt(input_data)   # encrypt file
endtime = time()
print('encrypt time {}s'.format(endtime - starttime))

with open("G:\method_test_save\AES\encrypted.png","wb") as w:
    w.write(enc_data)
w.close()

with open('G:\method_test_save\AES\encrypted.png','rb') as rf:
    input = rf.read()
rf.close()

cfb_decipher = AES.new(key, AES.MODE_CFB, iv)   # generate a decryptor with the same key
starttime = time()
decipher_data = cfb_decipher.decrypt(input) # decryptor
endtime = time()
print('decrypt time {}s'.format(endtime - starttime))

with open(r'G:\method_test_save\AES\decrypted.png','wb') as wf:
    wf.write(decipher_data)
wf.close()




def generate_key(str,key,iv):
    '''use cpu code to generate a private key for AES encryption'''
    cfb_cipher = AES.new(key, AES.MODE_CFB, iv) # generate a encryptor

    enc_str = cfb_cipher.encrypt(str)

    return enc_str

def encrypt_img(img,key,iv):
    '''encrypt img'''
    
    cfb_cipher = AES.new(key, AES.MODE_CFB, iv)
    enc_img = cfb_cipher.encrypt(img)
    return enc_img

def decrypt_img(enc_img,key,iv):
    '''decrypt img'''
    cfb_decipher = AES.new(key, AES.MODE_CFB, iv)
    dec_img = cfb_decipher.decrypt(enc_img)
    return dec_img


# if __name__ =="__main__":
#     cpuid = '0x0A'






