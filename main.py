from core.compose_music import Synthesizer
from core.load_model import train_model

if __name__ == '__main__':
    model, notes, seed = train_model()
    synth = Synthesizer(model, notes, seed)
    composed_notes, melody = synth.compose(100)
    melody.write('midi', 'Melody_Generated.mid')
