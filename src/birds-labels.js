// Labels for the zero-shot "Birds" mode. ImageNet-1k (used by the other two models) has only 59 bird
// species and lacks many common ones, e.g. kingfisher, pigeon, sparrow, myna, parakeet and crow, so a
// kingfisher can only ever be reported as its nearest ImageNet look-alike (jacamar). CLIP compares the
// photo with these names instead. Edit the lists and re-run scripts/precompute-bird-embeddings.mjs.
export const BIRD_PROMPT = (name) => `a photo of a ${name}, a type of bird.`

export const BIRDS = [
  // common in India
  'common kingfisher', 'white-throated kingfisher', 'pied kingfisher', 'stork-billed kingfisher',
  'house crow', 'large-billed crow', 'rock pigeon', 'spotted dove', 'laughing dove', 'Eurasian collared dove',
  'house sparrow', 'common myna', 'bank myna', 'jungle myna', 'rose-ringed parakeet', 'Alexandrine parakeet',
  'plum-headed parakeet', 'Indian peafowl', 'red-vented bulbul', 'red-whiskered bulbul', 'Asian koel',
  'greater coucal', 'black kite', 'brahminy kite', 'shikra', 'barn owl', 'spotted owlet', 'Indian roller',
  'green bee-eater', 'coppersmith barbet', 'oriental magpie-robin', 'Indian robin', 'purple sunbird',
  'black drongo', 'Indian pond heron', 'cattle egret', 'little egret', 'grey heron', 'painted stork',
  'Asian openbill', 'sarus crane', 'Eurasian hoopoe', 'red-wattled lapwing', 'Indian grey hornbill',
  'great hornbill', 'baya weaver', 'scaly-breasted munia', 'jungle babbler', 'rufous treepie', 'barn swallow',
  'oriental white-eye', 'common tailorbird', 'red junglefowl', 'domestic chicken',
  // well known elsewhere
  'jacamar', 'hummingbird', 'toucan', 'flamingo', 'penguin', 'ostrich', 'emu', 'bald eagle', 'golden eagle',
  'peregrine falcon', 'scarlet macaw', 'sulphur-crested cockatoo', 'budgerigar', 'cockatiel', 'lovebird',
  'American robin', 'European robin', 'blue jay', 'northern cardinal', 'American goldfinch', 'great tit',
  'blue tit', 'Eurasian magpie', 'mallard duck', 'mute swan', 'black swan', 'Canada goose', 'pelican',
  'seagull', 'Atlantic puffin', 'woodpecker', 'wild turkey', 'kiwi', 'snowy owl', 'great horned owl',
  'albatross', 'cormorant', 'kookaburra',
  // the remaining ImageNet-1k bird species, so this mode covers everything the other models know
  'rooster', 'brambling', 'European goldfinch', 'house finch', 'dark-eyed junco', 'indigo bunting',
  'chickadee', 'white-throated dipper', 'vulture', 'great grey owl', 'black grouse', 'ptarmigan',
  'ruffed grouse', 'prairie chicken', 'quail', 'partridge', 'African grey parrot', 'rainbow lorikeet',
  'red-breasted merganser', 'white stork', 'black stork', 'spoonbill', 'little blue heron', 'great egret',
  'bittern', 'limpkin', 'purple swamphen', 'American coot', 'bustard', 'ruddy turnstone', 'dunlin',
  'common redshank', 'dowitcher', 'oystercatcher', 'king penguin',
]

// Names that correspond to an ImageNet-1k bird class. For these the ImageNet model (ViT-B/16) is more
// accurate, so the page only uses a CLIP answer when it names a species outside this set.
export const IMAGENET_COVERED = new Set([
  'rooster', 'domestic chicken', 'ostrich', 'brambling', 'European goldfinch', 'house finch', 'dark-eyed junco',
  'indigo bunting', 'American robin', 'red-vented bulbul', 'red-whiskered bulbul', 'blue jay', 'Eurasian magpie',
  'chickadee', 'white-throated dipper', 'black kite', 'brahminy kite', 'bald eagle', 'vulture', 'great grey owl',
  'black grouse', 'ptarmigan', 'ruffed grouse', 'prairie chicken', 'Indian peafowl', 'quail', 'partridge',
  'African grey parrot', 'scarlet macaw', 'sulphur-crested cockatoo', 'rainbow lorikeet', 'greater coucal',
  'green bee-eater', 'Indian grey hornbill', 'great hornbill', 'hummingbird', 'jacamar', 'toucan', 'mallard duck',
  'red-breasted merganser', 'Canada goose', 'black swan', 'white stork', 'black stork', 'spoonbill', 'flamingo',
  'little blue heron', 'great egret', 'bittern', 'sarus crane', 'limpkin', 'purple swamphen', 'American coot',
  'bustard', 'ruddy turnstone', 'dunlin', 'common redshank', 'dowitcher', 'oystercatcher', 'pelican', 'king penguin',
  'albatross',
])

// Competing descriptions so that photos without a bird are reported as such.
export const NOT_BIRD_LABEL = 'Not a bird'
export const NOT_BIRD_PROMPTS = [
  'a photo of a person.', 'a photo of a mammal.', 'a photo of an insect.', 'a photo of a fish.',
  'a photo of a vehicle.', 'a photo of a building.', 'a photo of food.', 'a photo of an object.',
  'a photo of a plant or flower.', 'a photo of a landscape.',
]
