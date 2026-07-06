"""
Generate draf jurnal Word (.docx) dari hasil penelitian MobileNetV2
"""
import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

# ── Margin ──────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin   = Inches(1.25)
    section.right_margin  = Inches(1.25)

# ── Helper ───────────────────────────────────────────────────────
def heading(text, level=1):
    h = doc.add_heading(text, level)
    h.runs[0].font.color.rgb = RGBColor(0, 0, 0)
    return h

def para(text, bold=False, italic=False, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    if align:
        p.alignment = align
    return p

def table_with_header(header_row, data_rows):
    t = doc.add_table(rows=1 + len(data_rows), cols=len(header_row))
    t.style = 'Light Grid Accent 1'
    hdr = t.rows[0].cells
    for i, h in enumerate(header_row):
        hdr[i].text = h
        hdr[i].paragraphs[0].runs[0].bold = True
    for r, row in enumerate(data_rows, 1):
        for c, val in enumerate(row):
            t.rows[r].cells[c].text = str(val)
    doc.add_paragraph()
    return t

# ════════════════════════════════════════════════════════════════
# JUDUL & INFO
# ════════════════════════════════════════════════════════════════
t = doc.add_heading(
    'KLASIFIKASI CITRA BENCANA ALAM MENGGUNAKAN MOBILENETV2 BERBASIS DEEP LEARNING',
    0
)
t.alignment = WD_ALIGN_PARAGRAPH.CENTER

p = doc.add_paragraph()
r = p.add_run('[Nama Penulis]')
r.bold = True
p.alignment = WD_ALIGN_PARAGRAPH.CENTER

p2 = doc.add_paragraph('[Nama Universitas/Institusi], [Kota], Indonesia')
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER

p3 = doc.add_paragraph('[email@institusi.ac.id]')
p3.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph()

# ════════════════════════════════════════════════════════════════
# ABSTRAK
# ════════════════════════════════════════════════════════════════
heading('ABSTRAK', 1)
doc.add_paragraph(
    'Bencana alam seperti banjir, gempa bumi, kebakaran hutan, tanah longsor, dan kerusakan '
    'infrastruktur menimbulkan kerugian material dan korban jiwa yang sangat besar. Sistem '
    'pendeteksian bencana berbasis citra digital dapat mempercepat respons darurat. Penelitian '
    'ini mengusulkan model klasifikasi citra bencana alam menggunakan MobileNetV2 berbasis '
    'deep learning dengan teknik transfer learning dua fase, yaitu feature extraction dan '
    'fine-tuning. Dataset yang digunakan adalah Comprehensive Disaster Dataset (CDD) yang '
    'bersumber dari Kaggle, terdiri dari 13.557 citra yang dibagi menjadi 6 kelas: '
    'Damaged_Infrastructure, Fire_Disaster, Human_Damage, Land_Disaster, Non_Damage, dan '
    'Water_Disaster. Augmentasi data diterapkan untuk mengatasi ketidakseimbangan kelas. '
    'Data dibagi dengan rasio 70:15:15 untuk training, validasi, dan pengujian. '
    'Hasil evaluasi pada 2.040 citra uji menunjukkan akurasi sebesar 94,31%, precision 94,57%, '
    'recall 94,31%, F1-Score 94,25%, dan AUC-ROC (macro) sebesar 0,9924. Visualisasi '
    'Grad-CAM mengonfirmasi bahwa model berfokus pada fitur visual yang relevan secara semantik. '
    'Model MobileNetV2 dengan transfer learning terbukti efektif dan efisien untuk klasifikasi '
    'citra bencana alam.'
)
p = doc.add_paragraph()
p.add_run('Kata kunci: ').bold = True
p.add_run('klasifikasi citra, bencana alam, deep learning, MobileNetV2, transfer learning, Grad-CAM, CNN')

doc.add_page_break()

# ════════════════════════════════════════════════════════════════
# 1. PENDAHULUAN
# ════════════════════════════════════════════════════════════════
heading('1. PENDAHULUAN', 1)
doc.add_paragraph(
    'Indonesia merupakan salah satu negara paling rawan bencana alam di dunia akibat letak '
    'geografisnya yang berada di kawasan Cincin Api Pasifik (Ring of Fire). Bencana alam '
    'seperti gempa bumi, tsunami, banjir, tanah longsor, dan kebakaran hutan terjadi hampir '
    'setiap tahun dengan dampak yang sangat signifikan. Data Badan Nasional Penanggulangan '
    'Bencana (BNPB) mencatat ribuan kejadian bencana setiap tahunnya yang mengakibatkan '
    'kerugian ekonomi dan korban jiwa. Deteksi dini dan respons cepat terhadap bencana '
    'menjadi krusial untuk meminimalkan dampak tersebut.'
)
doc.add_paragraph(
    'Perkembangan teknologi computer vision dan deep learning telah membuka peluang baru '
    'dalam sistem deteksi dan klasifikasi bencana berbasis citra digital. Convolutional '
    'Neural Network (CNN) terbukti unggul dalam mengekstrak fitur visual kompleks dari citra. '
    'Namun, melatih CNN dari nol memerlukan dataset berukuran besar dan sumber daya komputasi '
    'yang tinggi. Transfer learning menjadi solusi praktis dengan memanfaatkan representasi '
    'fitur dari model yang telah dilatih pada dataset besar seperti ImageNet.'
)
doc.add_paragraph(
    'Beberapa penelitian terdahulu telah mengeksplorasi CNN untuk klasifikasi citra bencana, '
    'namun masih terdapat keterbatasan: penggunaan arsitektur berat seperti VGG dan ResNet '
    'yang tidak efisien untuk deployment, dataset yang terbatas, serta kurangnya interpretabilitas '
    'model. MobileNetV2 hadir sebagai alternatif arsitektur yang ringan namun kompetitif, '
    'dengan inverted residual blocks yang mengurangi parameter tanpa mengorbankan akurasi.'
)
doc.add_paragraph(
    'Penelitian ini bertujuan membangun model klasifikasi citra bencana alam menggunakan '
    'MobileNetV2 berbasis deep learning dengan pendekatan transfer learning dua fase pada '
    'Comprehensive Disaster Dataset (CDD). Selain itu, Grad-CAM digunakan untuk '
    'memvisualisasikan area penting yang difokuskan model dalam proses klasifikasi, '
    'sehingga meningkatkan kepercayaan dan interpretabilitas hasil.'
)

# ════════════════════════════════════════════════════════════════
# 2. TINJAUAN PUSTAKA
# ════════════════════════════════════════════════════════════════
heading('2. TINJAUAN PUSTAKA', 1)

heading('2.1 Deep Learning dan Convolutional Neural Network (CNN)', 2)
doc.add_paragraph(
    'Deep learning adalah cabang machine learning yang menggunakan neural network dengan '
    'banyak lapisan (layer) untuk mempelajari representasi data secara hierarkis. CNN '
    'dirancang khusus untuk data spasial seperti citra dengan menggunakan layer convolutional '
    'yang mengekstrak fitur dari pola lokal (tepi, tekstur, bentuk). Arsitektur CNN umumnya '
    'terdiri dari convolutional layer, pooling layer, batch normalization, activation function '
    '(ReLU), dan fully connected layer. LeCun et al. (1998) merupakan pelopor pengembangan '
    'CNN yang kemudian berkembang pesat setelah AlexNet memenangkan ImageNet Large Scale '
    'Visual Recognition Challenge (ILSVRC) pada tahun 2012.'
)

heading('2.2 Transfer Learning', 2)
doc.add_paragraph(
    'Transfer learning adalah teknik yang memanfaatkan pengetahuan (bobot) dari model yang '
    'telah dilatih pada task atau dataset lain untuk mempercepat pembelajaran pada task baru. '
    'Terdapat dua strategi utama dalam transfer learning: (1) Feature Extraction, di mana '
    'base model dibekukan (frozen) dan hanya lapisan klasifikasi baru yang dilatih; '
    '(2) Fine-Tuning, di mana sebagian layer atas base model di-unfreeze dan dilatih ulang '
    'dengan learning rate yang sangat kecil. Kombinasi kedua strategi ini terbukti memberikan '
    'hasil optimal dalam berbagai task klasifikasi citra.'
)

heading('2.3 MobileNetV2', 2)
doc.add_paragraph(
    'MobileNetV2 diperkenalkan oleh Sandler et al. (2018) sebagai arsitektur CNN yang '
    'dioptimalkan untuk perangkat mobile dan embedded. Inovasi utamanya adalah Inverted '
    'Residual Block dengan Linear Bottleneck yang menggunakan depthwise separable convolution '
    'untuk mengurangi jumlah operasi matematik. MobileNetV2 memiliki sekitar 3,5 juta '
    'parameter dengan ukuran model 14MB, namun mencapai top-1 accuracy 71,8% pada ImageNet. '
    'Karakteristik ini menjadikan MobileNetV2 ideal untuk sistem real-time yang memerlukan '
    'efisiensi komputasi tinggi.'
)

heading('2.4 Grad-CAM (Gradient-weighted Class Activation Mapping)', 2)
doc.add_paragraph(
    'Grad-CAM diperkenalkan oleh Selvaraju et al. (2017) sebagai teknik interpretabilitas '
    'untuk model deep learning berbasis CNN. Metode ini menghitung gradient dari skor kelas '
    'terhadap feature map pada layer konvolusi terakhir, kemudian melakukan global average '
    'pooling untuk menghasilkan bobot kepentingan setiap channel. Hasil heatmap ditumpangkan '
    'pada citra asli untuk memvisualisasikan area yang paling berkontribusi pada keputusan '
    'klasifikasi. Grad-CAM tidak memerlukan modifikasi arsitektur model dan dapat diaplikasikan '
    'pada arsitektur CNN apapun.'
)

heading('2.5 Penelitian Terkait', 2)
doc.add_paragraph(
    'Berbagai penelitian telah dilakukan untuk klasifikasi citra bencana menggunakan deep learning. '
    '[Sebutkan penelitian terkait yang kamu temukan di Google Scholar dan masukkan ke tabel di bawah].'
)
table_with_header(
    ['Peneliti', 'Tahun', 'Model', 'Dataset', 'Akurasi'],
    [
        ['[Penulis A]', '[Tahun]', '[Model]', '[Dataset]', '[XX%]'],
        ['[Penulis B]', '[Tahun]', '[Model]', '[Dataset]', '[XX%]'],
        ['[Penulis C]', '[Tahun]', '[Model]', '[Dataset]', '[XX%]'],
        ['Penelitian ini', '2024', 'MobileNetV2', 'CDD (13.557 citra)', '94,31%'],
    ]
)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════
# 3. METODOLOGI
# ════════════════════════════════════════════════════════════════
heading('3. METODOLOGI', 1)

heading('3.1 Dataset', 2)
doc.add_paragraph(
    'Penelitian ini menggunakan Comprehensive Disaster Dataset (CDD) yang tersedia di platform '
    'Kaggle. Dataset ini terdiri dari 13.557 citra digital yang dikumpulkan dari berbagai sumber '
    'terkait kejadian bencana alam dan kondisi normal. Dataset memiliki struktur subfolder '
    'berlapis yang kemudian dipetakan ke dalam 6 kelas utama. Distribusi kelas ditampilkan '
    'pada Tabel 1.'
)
table_with_header(
    ['No', 'Kelas', 'Jumlah Citra', 'Persentase (%)'],
    [
        ['1', 'Damaged_Infrastructure', '1.454', '10,7'],
        ['2', 'Fire_Disaster',          '933',   '6,9'],
        ['3', 'Human_Damage',           '241',   '1,8'],
        ['4', 'Land_Disaster',          '657',   '4,8'],
        ['5', 'Non_Damage',             '9.237', '68,1'],
        ['6', 'Water_Disaster',         '1.035', '7,6'],
        ['',  'Total',                  '13.557','100'],
    ]
)
doc.add_paragraph(
    'Tabel 1. Distribusi kelas pada Comprehensive Disaster Dataset (CDD).\n'
    'Catatan: Dataset menunjukkan ketidakseimbangan kelas (imbalanced) dengan kelas '
    'Non_Damage mendominasi 68,1% dari total data. Kondisi ini diatasi dengan teknik '
    'augmentasi data pada set training.'
)

heading('3.2 Preprocessing', 2)
doc.add_paragraph(
    'Tahap preprocessing meliputi: (1) validasi integritas citra menggunakan PIL dengan '
    'metode load() untuk mendeteksi file corrupt dan truncated — ditemukan 2 file rusak '
    'yang dihapus dari dataset; (2) resize citra ke ukuran 224×224 piksel sesuai input '
    'MobileNetV2; (3) normalisasi nilai piksel menggunakan fungsi preprocess_input dari '
    'tensorflow.keras.applications.mobilenet_v2 yang mengubah skala nilai piksel dari '
    '[0, 255] ke rentang [-1, 1], sesuai dengan spesifikasi pretrained MobileNetV2 pada ImageNet.'
)

heading('3.3 Augmentasi Data', 2)
doc.add_paragraph(
    'Augmentasi data diterapkan pada set training untuk meningkatkan keragaman data dan '
    'mencegah overfitting, terutama pada kelas dengan jumlah sampel sedikit. Augmentasi '
    'tidak diterapkan pada data validasi dan pengujian. Parameter augmentasi disajikan '
    'pada Tabel 2.'
)
table_with_header(
    ['Teknik Augmentasi', 'Parameter'],
    [
        ['Rotation range',          '20 derajat'],
        ['Width shift range',       '0,2 (20%)'],
        ['Height shift range',      '0,2 (20%)'],
        ['Shear range',             '0,15'],
        ['Zoom range',              '0,2 (20%)'],
        ['Horizontal flip',         'True'],
        ['Brightness range',        '[0,8 - 1,2]'],
        ['Fill mode',               'nearest'],
    ]
)
doc.add_paragraph('Tabel 2. Parameter augmentasi data pada set training.')

heading('3.4 Pembagian Dataset', 2)
doc.add_paragraph(
    'Dataset dibagi secara stratifikasi (per kelas) menggunakan rasio 70:15:15 untuk '
    'memastikan distribusi kelas yang proporsional pada setiap split. Pembagian dataset '
    'ditampilkan pada Tabel 3.'
)
table_with_header(
    ['Split', 'Jumlah Citra', 'Rasio (%)'],
    [
        ['Training',    '9.486', '70'],
        ['Validasi',    '2.031', '15'],
        ['Pengujian',   '2.040', '15'],
        ['Total',       '13.557','100'],
    ]
)
doc.add_paragraph('Tabel 3. Pembagian dataset (train/validasi/test).')

heading('3.5 Arsitektur Model', 2)
doc.add_paragraph(
    'Model yang dibangun terdiri dari dua bagian utama: (1) Base model MobileNetV2 pretrained '
    'ImageNet sebagai feature extractor; (2) Custom classification head untuk klasifikasi 6 kelas. '
    'Arsitektur lengkap model disajikan pada Tabel 4.'
)
table_with_header(
    ['Layer', 'Output Shape', 'Keterangan'],
    [
        ['Input',                    '(224, 224, 3)',  'Citra RGB'],
        ['MobileNetV2 (base)',        '(7, 7, 1280)',   'Pretrained ImageNet, frozen Phase 1'],
        ['GlobalAveragePooling2D',    '(1280,)',         'Reduksi spasial'],
        ['BatchNormalization',        '(1280,)',         'Normalisasi aktivasi'],
        ['Dense(256, ReLU)',          '(256,)',          'Feature learning'],
        ['Dropout(0.3)',              '(256,)',          'Regularisasi'],
        ['Dense(128, ReLU)',          '(128,)',          'Feature learning'],
        ['Dropout(0.3)',              '(128,)',          'Regularisasi'],
        ['Dense(6, Softmax)',         '(6,)',            'Output 6 kelas'],
    ]
)
doc.add_paragraph('Tabel 4. Arsitektur model DisasterMobileNetV2.')
doc.add_paragraph(
    'Total parameter model: 2.624.710, terdiri dari ~2.257.984 parameter frozen (MobileNetV2 base) '
    'dan ~366.726 parameter trainable (custom head) pada Phase 1.'
)

heading('3.6 Strategi Training Dua Fase', 2)
doc.add_paragraph(
    'Training dilakukan dalam dua fase untuk memaksimalkan transfer learning:'
)
doc.add_paragraph(
    'Phase 1 — Feature Extraction: Base model MobileNetV2 dibekukan (semua layer frozen). '
    'Hanya custom head yang dilatih selama 10 epoch dengan learning rate 1×10⁻³ menggunakan '
    'optimizer Adam dan loss categorical cross-entropy. Fase ini memungkinkan head beradaptasi '
    'dengan fitur CDD tanpa mengganggu bobot ImageNet.'
)
doc.add_paragraph(
    'Phase 2 — Fine-Tuning: Layer base model mulai dari layer ke-100 di-unfreeze dan dilatih '
    'bersama head selama 10 epoch dengan learning rate yang sangat kecil (1×10⁻⁴). '
    'Penggunaan LR kecil mencegah catastrophic forgetting pada fitur ImageNet yang sudah baik.'
)
table_with_header(
    ['Parameter', 'Phase 1 (Feature Extraction)', 'Phase 2 (Fine-Tuning)'],
    [
        ['Base model',    'Frozen',          'Unfreeze layer ≥ 100'],
        ['Epochs',        '10',              '10'],
        ['Learning Rate', '1×10⁻³',          '1×10⁻⁴'],
        ['Optimizer',     'Adam',            'Adam'],
        ['Loss',          'Cat. Crossentropy','Cat. Crossentropy'],
        ['Batch Size',    '32',              '32'],
    ]
)
doc.add_paragraph('Tabel 5. Hyperparameter training dua fase.')

heading('3.7 Callback Training', 2)
doc.add_paragraph(
    'Tiga callback digunakan selama training: (1) ModelCheckpoint — menyimpan bobot terbaik '
    'berdasarkan val_accuracy; (2) EarlyStopping — menghentikan training jika val_accuracy '
    'tidak meningkat selama 10 epoch; (3) ReduceLROnPlateau — mengurangi learning rate '
    'sebesar faktor 0,5 jika val_loss tidak menurun selama 5 epoch, dengan nilai minimum 1×10⁻⁷.'
)

heading('3.8 Metrik Evaluasi', 2)
doc.add_paragraph(
    'Model dievaluasi menggunakan lima metrik utama berdasarkan confusion matrix:'
)
doc.add_paragraph('Accuracy = (TP + TN) / (TP + TN + FP + FN)')
doc.add_paragraph('Precision = TP / (TP + FP)')
doc.add_paragraph('Recall    = TP / (TP + FN)')
doc.add_paragraph('F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)')
doc.add_paragraph(
    'AUC-ROC (Area Under Curve) dihitung secara One-vs-Rest (OvR) untuk setiap kelas, '
    'kemudian dirata-ratakan (macro average).'
)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════
# 4. HASIL DAN PEMBAHASAN
# ════════════════════════════════════════════════════════════════
heading('4. HASIL DAN PEMBAHASAN', 1)

heading('4.1 Kurva Training', 2)
doc.add_paragraph(
    'Gambar 1 menunjukkan kurva accuracy dan loss selama training dua fase. '
    'Pada Phase 1 (epoch 1-10), model konvergen dengan cepat karena base model '
    'MobileNetV2 sudah membawa representasi fitur yang kuat dari ImageNet. '
    'Val_accuracy terbaik Phase 1 mencapai 90,84%. '
    'Pada Phase 2 (epoch 11-20), fine-tuning meningkatkan val_accuracy hingga 94,98%. '
    'Garis putus-putus pada epoch ke-10 menandai transisi antara dua fase training.'
)
doc.add_paragraph('[Gambar 1: Kurva Training & Validation — Accuracy dan Loss (training_curves.png)]')
doc.add_paragraph(
    'Terdapat gap antara training accuracy (96,87%) dan validation accuracy (70,84%) pada epoch '
    'terakhir, namun model checkpoint menyimpan bobot terbaik sehingga hasil evaluasi test '
    'menggunakan bobot terbaik (val_accuracy = 94,98%).'
)

heading('4.2 Hasil Evaluasi pada Test Set', 2)
doc.add_paragraph(
    'Evaluasi dilakukan pada 2.040 citra test yang tidak pernah dilihat model selama training. '
    'Hasil evaluasi lengkap disajikan pada Tabel 6.'
)
table_with_header(
    ['Metrik', 'Nilai', 'Keterangan'],
    [
        ['Accuracy',          '94,31%', 'Overall correctness'],
        ['Precision',         '94,57%', 'Weighted average'],
        ['Recall',            '94,31%', 'Weighted average'],
        ['F1-Score',          '94,25%', 'Weighted average'],
        ['AUC-ROC (macro)',   '0,9924', 'One-vs-Rest, 6 kelas'],
        ['Loss',              '0,2236', 'Categorical cross-entropy'],
    ]
)
doc.add_paragraph('Tabel 6. Hasil evaluasi model pada test set.')
doc.add_paragraph(
    'Nilai AUC-ROC sebesar 0,9924 menunjukkan kemampuan diskriminasi model yang sangat baik '
    'untuk membedakan antar kelas bencana. Accuracy 94,31% mengindikasikan bahwa model '
    'berhasil mengklasifikasikan 1.924 dari 2.040 citra uji dengan benar.'
)

heading('4.3 Classification Report per Kelas', 2)
doc.add_paragraph(
    'Tabel 7 menyajikan metrik evaluasi per kelas bencana pada test set.'
)
table_with_header(
    ['Kelas', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Support'],
    [
        ['Damaged_Infrastructure', '0,7519', '0,9269', '0,8303', '0,9910', '219'],
        ['Fire_Disaster',          '0,9769', '0,9007', '0,9373', '0,9956', '141'],
        ['Human_Damage',           '0,9355', '0,7838', '0,8529', '0,9978', '37'],
        ['Land_Disaster',          '0,7763', '0,5900', '0,6705', '0,9819', '100'],
        ['Non_Damage',             '0,9935', '0,9942', '0,9939', '0,9996', '1.387'],
        ['Water_Disaster',         '0,8759', '0,8141', '0,8439', '0,9888', '156'],
        ['Macro avg',              '0,8850', '0,8350', '0,8548', '0,9924', '2.040'],
        ['Weighted avg',           '0,9457', '0,9431', '0,9425', '-',      '2.040'],
    ]
)
doc.add_paragraph('Tabel 7. Classification Report per kelas pada test set.')
doc.add_paragraph(
    'Kelas Non_Damage memperoleh F1-Score tertinggi (0,9939) karena jumlah sampel paling banyak '
    '(9.237 citra training). Kelas Land_Disaster memiliki F1-Score terendah (0,6705) karena '
    'jumlah sampel terbatas (657 citra) dan kemiripan visual dengan kelas lain. '
    'Kelas Fire_Disaster memperoleh precision tertinggi (0,9769) menunjukkan model sangat '
    'akurat saat memprediksi kelas ini.'
)

heading('4.4 Confusion Matrix', 2)
doc.add_paragraph(
    '[Gambar 2: Confusion Matrix (confusion_matrix.png)]\n'
    '[Gambar 3: Normalized Confusion Matrix (confusion_matrix_norm.png)]'
)
doc.add_paragraph(
    'Dari confusion matrix, terlihat bahwa kelas Non_Damage memiliki true positive rate '
    'tertinggi (99,42%). Kesalahan klasifikasi terbesar terjadi pada kelas Land_Disaster '
    'yang sering tertukar dengan kelas lain, kemungkinan disebabkan kemiripan tekstur '
    'citra tanah longsor dengan kerusakan infrastruktur. Kelas Damaged_Infrastructure '
    'memiliki recall tinggi (0,9269) namun precision lebih rendah (0,7519), '
    'artinya model cenderung over-predict kelas ini.'
)

heading('4.5 ROC-AUC Curve', 2)
doc.add_paragraph('[Gambar 4: ROC-AUC Curve per kelas (roc_curves.png)]')
doc.add_paragraph(
    'Semua kelas menunjukkan nilai AUC di atas 0,98 kecuali Land_Disaster (0,9819) dan '
    'Water_Disaster (0,9888). Nilai AUC mendekati 1,0 pada Non_Damage (0,9996) dan '
    'Human_Damage (0,9978). Secara keseluruhan, macro AUC sebesar 0,9924 '
    'mengindikasikan kemampuan diskriminasi yang sangat baik pada semua kelas.'
)

heading('4.6 Visualisasi Grad-CAM', 2)
doc.add_paragraph('[Gambar 5: Grad-CAM Grid — semua kelas (gradcam_grid.png)]')
doc.add_paragraph(
    'Visualisasi Grad-CAM menunjukkan bahwa model MobileNetV2 berfokus pada fitur visual '
    'yang secara semantik relevan untuk setiap kelas:'
)
items = [
    'Damaged_Infrastructure: model fokus pada area retakan, reruntuhan, dan struktur bangunan rusak.',
    'Fire_Disaster: model mendeteksi area berwarna oranye-merah (api) dan asap.',
    'Human_Damage: area kerumunan manusia dan aktivitas evakuasi menjadi fokus model.',
    'Land_Disaster: model fokus pada perubahan kontur tanah dan area longsor.',
    'Non_Damage: bangunan utuh, vegetasi hijau, dan permukaan air yang tenang.',
    'Water_Disaster: genangan air coklat dan area banjir menjadi highlight utama.',
]
for item in items:
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(item)

doc.add_paragraph(
    'Hasil Grad-CAM mengonfirmasi bahwa model tidak hanya menghafal pola statistik dataset, '
    'tetapi benar-benar belajar fitur visual yang meaningful untuk identifikasi bencana. '
    'Ini meningkatkan kepercayaan terhadap model untuk aplikasi praktis.'
)

heading('4.7 Perbandingan dengan Penelitian Terdahulu', 2)
doc.add_paragraph(
    'Tabel 8 membandingkan hasil penelitian ini dengan beberapa penelitian terkait '
    'di bidang klasifikasi citra bencana. [Lengkapi tabel dengan referensi aktual dari literatur].'
)
table_with_header(
    ['Peneliti', 'Tahun', 'Model', 'Dataset', 'Accuracy'],
    [
        ['[Penulis A]',   '[Tahun]', 'VGG16',    '[Dataset]',       '[XX%]'],
        ['[Penulis B]',   '[Tahun]', 'ResNet50',  '[Dataset]',       '[XX%]'],
        ['[Penulis C]',   '[Tahun]', 'InceptionV3','[Dataset]',      '[XX%]'],
        ['Penelitian ini','2024',    'MobileNetV2','CDD (13.557)',    '94,31%'],
    ]
)
doc.add_paragraph('Tabel 8. Perbandingan dengan penelitian terdahulu.')

doc.add_page_break()

# ════════════════════════════════════════════════════════════════
# 5. KESIMPULAN
# ════════════════════════════════════════════════════════════════
heading('5. KESIMPULAN', 1)
doc.add_paragraph(
    'Penelitian ini berhasil membangun model klasifikasi citra bencana alam menggunakan '
    'MobileNetV2 berbasis deep learning dengan teknik transfer learning dua fase. Evaluasi '
    'pada 2.040 citra test dari Comprehensive Disaster Dataset (CDD) menunjukkan performa '
    'yang sangat baik dengan akurasi 94,31%, F1-Score 94,25%, dan AUC-ROC (macro) 0,9924. '
    'Strategi fine-tuning meningkatkan validation accuracy dari 90,84% (Phase 1) menjadi '
    '94,98% (Phase 2), membuktikan efektivitas pendekatan dua fase.'
)
doc.add_paragraph(
    'Visualisasi Grad-CAM mengonfirmasi bahwa model berfokus pada fitur visual yang secara '
    'semantik relevan untuk setiap kategori bencana, seperti api dan asap pada Fire_Disaster, '
    'genangan air pada Water_Disaster, dan retakan pada Damaged_Infrastructure. Hal ini '
    'meningkatkan interpretabilitas dan kepercayaan terhadap keputusan model.'
)
doc.add_paragraph(
    'Penelitian ini memberikan kontribusi berupa implementasi MobileNetV2 yang efisien untuk '
    'klasifikasi citra bencana dengan akurasi kompetitif, cocok untuk deployment pada sistem '
    'real-time dan perangkat dengan komputasi terbatas. Model yang dihasilkan dapat menjadi '
    'dasar sistem deteksi bencana otomatis yang mendukung upaya mitigasi dan respons darurat.'
)

heading('5.1 Keterbatasan Penelitian', 2)
doc.add_paragraph(
    'Beberapa keterbatasan yang ditemukan: (1) Dataset mengalami ketidakseimbangan kelas '
    'dengan Non_Damage mendominasi 68,1%, meski telah diatasi dengan augmentasi; '
    '(2) Kelas Land_Disaster memiliki performa terendah (F1: 0,6705) akibat keterbatasan '
    'jumlah sampel dan kemiripan visual dengan kelas lain; (3) Model belum diuji pada '
    'citra real-time dari drone atau satelit; (4) Evaluasi hanya menggunakan metrik standar, '
    'belum mempertimbangkan aspek waktu inferensi dan efisiensi memori pada perangkat mobile.'
)

heading('5.2 Saran untuk Penelitian Lanjutan', 2)
items_saran = [
    'Menambah jumlah sampel pada kelas minoritas (Human_Damage, Land_Disaster) melalui '
    'pengumpulan data tambahan atau teknik oversampling berbasis GAN.',
    
    'Mengeksplorasi arsitektur lain yang lebih baru seperti EfficientNetV2, ConvNeXt, atau '
    'Vision Transformer (ViT) untuk perbandingan performa.',
    
    'Menerapkan teknik ensemble learning dengan kombinasi beberapa model untuk meningkatkan '
    'robustness prediksi.',
    
    'Mengembangkan model multi-label classification untuk mendeteksi kombinasi bencana '
    'dalam satu citra (misalnya: banjir dan kerusakan infrastruktur secara bersamaan).',
    
    'Melakukan evaluasi pada data real-time dari UAV (Unmanned Aerial Vehicle) atau '
    'citra satelit untuk menguji generalisasi model pada kondisi nyata.',
    
    'Mengoptimasi model dengan quantization dan pruning untuk deployment pada perangkat '
    'edge computing dengan keterbatasan komputasi.',
    
    'Mengintegrasikan model ke dalam sistem informasi geografis (GIS) untuk pemetaan '
    'distribusi bencana secara spasial.',
]
for saran in items_saran:
    p = doc.add_paragraph(style='List Number')
    p.add_run(saran)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════
# DAFTAR PUSTAKA
# ════════════════════════════════════════════════════════════════
heading('DAFTAR PUSTAKA', 1)
doc.add_paragraph(
    '[1] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). '
    'MobileNetV2: Inverted Residuals and Linear Bottlenecks. In Proceedings of the IEEE '
    'Conference on Computer Vision and Pattern Recognition (CVPR), 4510-4520.'
)
doc.add_paragraph(
    '[2] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). '
    'Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. '
    'In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 618-626.'
)
doc.add_paragraph(
    '[3] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). '
    'Gradient-Based Learning Applied to Document Recognition. '
    'Proceedings of the IEEE, 86(11), 2278-2324.'
)
doc.add_paragraph(
    '[4] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). '
    'ImageNet Classification with Deep Convolutional Neural Networks. '
    'In Advances in Neural Information Processing Systems (NeurIPS), 1097-1105.'
)
doc.add_paragraph(
    '[Tambahkan 6-10 referensi lainnya dari literatur terkait klasifikasi citra bencana '
    'yang kamu temukan di Google Scholar, IEEE Xplore, atau Springer Link]'
)

# ════════════════════════════════════════════════════════════════
# SAVE DOCUMENT
# ════════════════════════════════════════════════════════════════
os.makedirs('paper', exist_ok=True)
output_path = 'paper/Jurnal_Klasifikasi_Bencana_MobileNetV2.docx'
doc.save(output_path)
print(f'✅ Draf jurnal berhasil dibuat!')
print(f'📄 File tersimpan di: {output_path}')
print()
print('Langkah selanjutnya:')
print('1. Buka file Word dan masukkan gambar dari folder results/')
print('2. Lengkapi tabel perbandingan penelitian dengan referensi aktual dari Google Scholar')
print('3. Isi nama penulis, afiliasi, dan email')
print('4. Tambahkan 10-15 referensi di Daftar Pustaka')
print('5. Format sesuai template jurnal target (IEEE/Springer/SINTA)')
print('6. Review grammar dan konsistensi penulisan')
