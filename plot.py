import matplotlib as plt
from data_preparation import train, val, unlabeled

counts = {
    'train': len(train),
    'val': len(val),
    'test': len(unlabeled)
}


plt.bar(counts.keys(), counts.values())
plt.title('Dataset Split Overview')
plt.xlabel('Dataset')
plt.ylabel('Number of Samples')

for i, v in enumerate(counts.values()):
    plt.text(i, v + 1, str(v), ha='center', fontsize=10)

plt.tight_layout()
plt.show()
