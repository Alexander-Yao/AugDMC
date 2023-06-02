from model import *
from torch import nn
from torchvision import transforms, datasets
import torch.multiprocessing
from parse import get_parse
from sklearn.metrics import accuracy_score, normalized_mutual_info_score as nmi, rand_score as ri, adjusted_rand_score as ar
from sklearn.cluster import KMeans
import os
import time

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    args = get_parse()
    clustering_times = 100
    this_seed = args.seed
    np.random.seed(this_seed)
    torch.manual_seed(this_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(this_seed)
    cnn = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=105, tau=args.tau)
    this_datetime = time.strftime("%m-%d-%Hh%Mm%Ss")

    normalize = transforms.Normalize(mean=[0.49159607, 0.44503418, 0.3952382],
                                     std=[0.26847756, 0.25570816, 0.25486323])

    data_transfrom = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        normalize
    ])
    if args.type == 'species':
        data_transfrom = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ColorJitter(hue=0.5),
            transforms.ColorJitter(contrast=0.5),
            transforms.ToTensor(),
            normalize
        ])
    elif args.type == 'color':
        data_transfrom = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
    elif args.type == 'grey':
        data_transfrom = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize
        ])
    i = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn.to(device)
    loss_function = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(cnn.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    img = datasets.ImageFolder(root='datasets/fruit', transform=data_transfrom)
    imgLoader = torch.utils.data.DataLoader(img, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # train
    cnn.train()
    best_loss, best_acc, last_update_epoch = np.inf, 0., 0
    for epoch in range(args.epoch):
        label_list = []
        predict_list = []
        loss_score = 0.
        for batch_index, (images, labels) in enumerate(imgLoader):
            label_list.extend(labels.cpu().numpy())

            outputs = cnn(images.to(device))
            predicted = torch.argmax(outputs.data, dim=1)
            optimizer.zero_grad()
            loss = loss_function(outputs, labels.to(device))
            loss_score += loss.cpu().item()

            loss.backward()
            optimizer.step()

            predict_list.extend(predicted.cpu().numpy())

        acc_score = accuracy_score(label_list, predict_list)
        print(f'Training epoch {epoch} loss: {loss_score:.3f} acc: {acc_score:.4f}')

        if epoch - last_update_epoch > 100:
            print('EarlyStop')
            break
        if acc_score > best_acc:
            last_update_epoch = epoch
            best_acc = acc_score
        if loss_score < best_loss:
            last_update_epoch = epoch
            best_loss = loss_score

    cnn.eval()
    res_path = os.path.join('res', args.dataset)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    if args.save:
        embedding_path = os.path.join('embedding', args.dataset)
        embedding_path = os.path.join(embedding_path, this_datetime)
        if not os.path.exists(embedding_path):
            os.makedirs(embedding_path)

    img_save = datasets.ImageFolder(root='datasets/fruit_type/color', transform=data_transfrom)
    imgLoader_save = torch.utils.data.DataLoader(img_save, batch_size=1, shuffle=False, num_workers=1)


    label_list = []
    embedding_list = []
    for batch_index, (images, labels) in enumerate(imgLoader_save):
        outputs = cnn.get_embedding(images.to(device))
        embedding_list.extend(outputs.cpu().detach().numpy())
        label_list.extend(labels.cpu().numpy())
    saved_embeddings = np.array(embedding_list)
    saved_labels = np.array(label_list)

    color_nmi_score = []
    color_ar_score = []
    color_ri_score = []
    for i in range(clustering_times):
        kmeans = KMeans(n_clusters=3, random_state=i).fit(saved_embeddings)
        pred_res = kmeans.labels_
        color_nmi_score.append(nmi(saved_labels, pred_res))
        color_ar_score.append(ar(saved_labels, pred_res))
        color_ri_score.append(ri(saved_labels, pred_res))

    img_save = datasets.ImageFolder(root='datasets/fruit_type/species', transform=data_transfrom)
    imgLoader_save = torch.utils.data.DataLoader(img_save, batch_size=1, shuffle=False, num_workers=1)

    label_list = []
    embedding_list = []
    for batch_index, (images, labels) in enumerate(imgLoader_save):
        outputs = cnn.get_embedding(images.to(device))
        embedding_list.extend(outputs.cpu().detach().numpy())
        label_list.extend(labels.cpu().numpy())
    saved_embeddings = np.array(embedding_list)
    saved_labels = np.array(label_list)


    species_nmi_score = []
    species_ar_score = []
    species_ri_score = []
    for i in range(clustering_times):
        kmeans = KMeans(n_clusters=3, random_state=i).fit(saved_embeddings)
        pred_res = kmeans.labels_
        species_nmi_score.append(nmi(saved_labels, pred_res))
        species_ar_score.append(ar(saved_labels, pred_res))
        species_ri_score.append(ri(saved_labels, pred_res))
