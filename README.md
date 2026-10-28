\# MNIST ile MLP Tabanlı El Yazısı Rakam Tanıma



MNIST veri seti üzerinde sıfırdan eğitilmiş Multi-Layer Perceptron (MLP) modeli. CNN ve hazır eğitilmiş model kullanılmamıştır. Kütüphane: PyTorch (CPU).



\## Veri ve ayrım

\- MNIST: 28x28 gri seviyeli, 10 sınıf.

\- Train / Validation / Test = 50.000 / 10.000 / 10.000 (train ve validation, orijinal 60.000'lik eğitim setinden `random\_split` ile, seed=42; test seti orijinal 10.000'lik test seti).

\- Normalizasyon: mean=0.1307, std=0.3081.

\- Test seti model geliştirme sürecinde kullanılmadı, konfigürasyon seçimi validation accuracy'ye göre yapıldı.



\## Ortak hiperparametreler

Batch size 64, 10 epoch, loss: CrossEntropyLoss, aktivasyon: ReLU, seed: 42.



\## Deney 1 – Temel MLP

784 → 256 → 128 → 10, Adam, lr=0.001.



| Metrik | Değer |

|---|---|

| Training loss | 0.0219 |

| Validation loss | 0.1171 |

| Training accuracy | 0.9933 |

| Validation accuracy | 0.9751 |

| Test accuracy | 0.9791 (test loss 0.0946) |

| Eğitim süresi | \~216 sn (CPU) |



Grafik: `deney1\_grafikler.png`



\## Hiperparametre karşılaştırması (Deney 1-5, son epoch, validation'a göre)



| Deney | Ayar | Train Loss | Val Loss | Train Acc | Val Acc | Süre (sn) |

|---|---|---|---|---|---|---|

| 1 | 256-128, Adam, lr=0.001 | 0.0219 | 0.1171 | 0.9933 | 0.9751 | 216.3 |

| 2 | 512-256, Adam, lr=0.001 | 0.0222 | 0.1367 | 0.9927 | 0.9732 | 226.3 |

| 3 | 256-128-64, Adam, lr=0.001 | 0.0316 | 0.1342 | 0.9896 | 0.9679 | 215.9 |

| 4 | 256-128, Adam, lr=0.01 | 0.0860 | 0.1887 | 0.9765 | 0.9598 | 215.2 |

| 5 | 256-128, SGD (momentum 0.9), lr=0.01 | 0.0049 | 0.0831 | 0.9992 | 0.9772 | 205.8 |



Seçilen model: Deney 5 (en yüksek validation accuracy ve en düşük validation loss). Deney 1 ile aradaki fark küçüktür (10.000 örnekte yaklaşık 21 örnek) ve tek seed ile çalışıldığı için kesin bir üstünlük olarak değerlendirilmemelidir.



\## Final model test sonucu (Deney 5)

Test loss 0.0693, \*\*test accuracy 0.9800\*\*. Test seti yalnızca bu seçim yapıldıktan sonra ölçülmüştür.



\## Kendi el yazım ile test

Öğrenci numaram `23120205050`, her hane ayrı ayrı yazılıp fotoğraflanmış ve `my\_digits/` içine konmuştur. Önişleme: gri tonlama, gerekirse renk ters çevirme (MNIST gibi koyu zemin, açık rakam), zemin temizleme, rakamın kırpılıp 20x20'ye sığdırılması, kütle merkezine göre 28x28 tuvale ortalanması, MNIST ile aynı mean/std normalizasyonu.



Sonuç: \*\*5/11 doğru\*\*. Ayrıntılar: `my\_digits\_sonuc.md`, görsel: `my\_digits\_sonuc.png`.



\## Genel Değerlendirme

Model MNIST test setinde %98.00 doğruluk elde ederken, kendi el yazımla hazırladığım 11 haneli öğrenci numarasında 5/11 doğru tahmin yaptı. Bu fark, eğitim verisi ile gerçek dünya verisi arasındaki dağılım farkını (distribution shift) gösteriyor. MNIST rakamları belirli bir yazım stiliyle, kalın ve yumuşak çizgilerle, ortalanmış olarak hazırlanmış. Benim görsellerim ise kağıda kalemle yazılıp fotoğraflandı ve çizgileri MNIST’e göre daha ince ve taneli.



Hataların bazıları rakamın yazılış biçiminden kaynaklanıyor olabilir: tabanlı bir 1 (MNIST’te 1'ler çoğunlukla düz çizgi) ve üstü düz yazılmış bir 3. Üç 0'ın yanlış tahmin edilmesi ise halkaların kesik ve ince olmasıyla ilişkili olabilir, ancak bunu ayrıca test etmedim. MLP girdiyi 784 bağımsız piksel olarak gördüğü için çizgi kalınlığı ve konum gibi küçük değişimlere hassas olabilir.



Önişlemede iki sorun yaşadım ve bunları ayırt ettim: (1) bazı görsellerde rakam dışı bir artefakt (üstte çubuk) vardı ve model onu 5'in parçası gibi okuyordu. Bu bir veri hazırlama hatasıydı, görselleri yeniden kırparak düzelttim. (2) Düzeltmeden sonra kalan hatalar gerçek genelleme hatası. Bu hataları düzeltmek için modeli yeniden eğitmedim, kendi el yazımı eğitim veya seçim sürecinde kullanmadım.



Sonuç olarak yalnızca 11 örnekle kesin bir doğruluk oranı söylemek mümkün değil, ama MNIST test başarısının gerçek el yazısına otomatik olarak aktarılmadığı net olarak görüldü. Sonuçları iyileştirmek için veri artırma (kaydırma, döndürme, kalınlık değişimi), kendi el yazısı örnekleriyle eğitim veya CNN gibi uzamsal yapıyı kullanan modeller denenebilir (bu ödevde CNN yasak olduğu için denenmedi).



\## Çalıştırma

```

pip install -r requirements.txt

python mnist\_mlp.py          # Deney 1-5 (yaklaşık 20 dk, CPU)

python final\_model.py        # final model + test

python predict\_my\_digits.py  # el yazısı tahmini (mlp\_final.pt gerekir)

```

