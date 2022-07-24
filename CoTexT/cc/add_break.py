import sentencepiece_model_pb2 as pb2

mp = pb2.ModelProto()
mp.ParseFromString(open("./sentencepiece.model", 'rb').read())
for i, sym in enumerate(["{", "}"], 1):
    new_sym = mp.SentencePiece()
    new_sym.piece = sym 
    new_sym.score = 0.0 # default score for USER_DEFINED
    new_sym.type = 4 # type value for USER_DEFINED
    mp.pieces.insert(-1, new_sym) # position after default control symbols ("<unk>", "<s>", "</s>")
outfile = './spm.model'
with open(outfile, 'wb') as f:
    f.write(mp.SerializeToString())