# DP4-Chord-Recognition
CSC475 Group Project - Chord Recognition from Audio

Group Members:
Aj Gray,
Matthew Reilly,
Shihong Li


Project for automatic chord recognition from audio.


# Introduction: 
Automatic chord recognition requires identification of chords present in a musical recording, and their occurrence intervals. It has a form similar to this (C major 0.0–2.3 s, A minor 2.3–4.6 s). The main application areas include chord visualizer programs, music education and analysis. In this project we will develop a chord recognition system, step-by-step, starting from simple and known approaches and adding improvements one by one, and evaluating the progress after each step.


# Problem statement: 
Main objective: build a system that takes an audio file as input and gives output in the form of a time-aligned series of chords.
Secondary objectives:

- build a rule-based baseline system based on chords templates;
  
- build a machine learning classifier and compare it with the baseline;

- enhance training data with data augmentation techniques (CQT rotation and audio pitch shifting), compare their efficiency;

- perform temporal smoothing of predictions with hidden Markov Model (HMM);

- assess the efficiency of the system with more difficult music genre (jazz) and compare human annotators' performance.


# Motivation: 
Chords provide the harmonic base in the majority of pop songs but transcription of them by hearing is quite hard and requires lots of training. Automatic systems will provide an easy way to transcript chords and analyze large collections of music. Also, this paradigm is easier to learn compared to the other MIR tasks. It has already established datasets, metrics and methods.


# Dataset description: 
Isophonics Beatles dataset: 180 songs from 12 Beatles albums with chords annotation and corresponding timestamps. This is an audio dataset which will be used mainly in our project.
CASD (Chordify Annotator Subjectivity Dataset): 50 songs annotated by 4 different annotators each time. The second dataset will be used to evaluate the amount of disagreement between annotators.


# Proposed methodology: 
Processing consists of the following pipeline:
Song audio → constant-Q transform (CQT) saying which notes are present → chord classifier that determines ambiguous chord → hidden Markov model (HMM) that performs temporal smoothing → final chords series.
The development will happen in order specified below.

  - Template baseline: compare chroma features with 24 major/minor chord templates plus a “no chord” class;
   
  - machine learning baseline: train a classifier with CQT features;

  - data augmentation via CQT rotation: rotate the CQT to get the same piece transposed to different key;
   
  - data augmentation via audio pitch shifting: change the pitch of the audio and compare with description in point 3.
   
  - HMM: apply temporal smoothing to predictions in order to prevent unrealistic fast chord changes;
   
  - extensions if possible: learnable pitch-class features, convolutional recurrent neural networks (CRNNs), expanded set of chord classes, jazz repertory;


# Tools and technologies: 
- python
- librosa (audio loading, CQT, chroma calculation, pitch shifting)
- mir_eval (chord evaluation)
- PyTorch (deep neural networks)
- NumPy, pandas, matplotlib
- pitchclass_cnn (pretrained pitch-class features)
- GitHub


# Evaluation metrics (Matthew): 
   
- Evaluation will be determined by a percentage based on the total number of correct chord identifications and the total number of chords designated in the .lab chord file.
  
- Evaluation may also be based on the variance of missed chords and consistency of missed chords for a specific chord or chord type.
  
- Evaluation will also be determined by length of correct segment identification.


# Timeline: 
- Week 1 (Oct 19 – 25)
  Set up the data and chord mapping
  Make the CQT and chord templates
  Load the data into PyTorch

- Week 2 (Oct 26 – Nov 1)
  Build the score checker
  Finish the template baseline
  Train a simple classifier

- Week 3 (Nov 2 – 8)
  Train the CNN and compare it with the template baseline
  Start the HMM smoother
  Make the comparison plots
  
- Week 4 (Nov 9 – 15)
  Add CQT rotation
  Finish the HMM smoother
  Make the confusion matrix
  
- Week 5 (Nov 16 – 22)
  Add pitch shifting and compare both augmentation methods
  Apply the HMM to the machine learning version

- Week 6 (Nov 23 – 30)
  Extensions if there is time
  Final results and report

# Individual objectives: 
PI – Performance indicator. Each objective consists of five PI values varying from basic to advanced.

### Member A: Matthew Reilly
Role: Data acquisition and score accuracy testing, verification and comparison
Objective 1: Build a chord database from audio files for identification and comparison for use by the baseline template and machine learning programs

- PI1 (basic): create a dictionary mapping complex chord values into the 12 primary major/minor chords or designate as no chord.
  
- PI2 (basic): process the .lab files and produce a new .lab with the simpler chord mappings
  
- PI3 (expected): Use librosa to detect and remove silence from audio file for timestamp alignment with the .lab chord file timestamps
  
- PI4 (expected): write script to process and pair .wav files to their associated simplified .lab file and produce a master .csv or .json file containing all relevant info into a single runnable file that can be given to the template or machine learning programs
  
- PI5 (advanced): expand chord mapping to include more complex chords for identification

Objective 2: Build a score checker to test the accuracy of the baseline template and machine learning program results to the database and produce comparison plots for easy visual verification of results

- PI1 (basic): split master file into a test file and a training file to ensure the machine learning program isn’t tested with files it was trained on skewing the results.
  
- PI2 (basic): write script to calculate accuracy percentage
  
- PI3 (expected): write a script using mir_eval to compare the generated chord predictions to the true chords from the .lab files for a given audio file
  
- PI4 (expected): create energy plots for audio sample with timestamped expected chords and timestamped identified chords for visual comparison of results
  
- PI5 (advanced): plot a confusion matrix showing which chords are misclassified most frequently
  

### Member B: AJ Gray
Role: Baseline template and Hidden-Markov Model Smoothing
Objective 1: Implement a functional chord recognition template version to compete against the machine learning version

- Write all templates (basic)
  
- Use librosa to make the CQT, which shows which notes are present (basic)
  
- Establish a rules-based system of chord recognition, ex: “C + E + G = C Major” (expected)
  
- Create baseline digital signal processing system based on the templates (expected)
  
- Get scores from the template version for the machine learning version to beat (expected)
  
Objective 2: Build an HMM smoother

- Research possible ways to program and implement a hidden-markov model smoother (basic)
  
- Work on a smoother that using the hidden-markov model to change poor guesses like C C G C C into smoother ones like C C C C C (expected)
  
- Use smoother with template version’s guesses and hand smoother over to machine learning model. (expected)
  
- Confirm whether or not the smoother is improving scores. If not, make sure it is working properly. (expected)
  
- Attempt to optimize the Hidden-Markov Method to improve scores further (advanced)

  
### Member C: Shihong Li
Role: Machine learning and data augmentation (steps 2, 3, 4)
Objective 1: Build machine learning classifier that determines chords based on Constant-Q Transform (CQT)

- PI1 (basic): Load CQT features and their chord labels for the whole set of Beatles' songs in Pytorch.
  
- PI2 (basic): Train a simple classifier (e.g. logistic regression or a multilayer perceptron) and get chord predictions.
  
- PI3 (expected): Train a small CNN classifier and test it using five folds cross-validation on albums level using mir_eval.
  
- PI4 (expected): Evaluate major/minor and root score performances of CNN and provide table comparison with the template baseline.
  
- PI5 (advanced): Evaluate how the window size and the network architecture complexity affect performance and find out best configurations.
  
Objective 2: Implement and analyze two data augmentation methods

- PI1 (basic): Implement a CQT rotation and corresponding shifts of labels.
  
- PI2 (basic): Generate the plots showing a CQT before rotation and a CQT after rotation along with corresponding labels to make sure that labels and features were shifted in the right way.
  
- PI3 (expected): Implement audio pitch shifting with the identical semitone range.
  
- PI4 (expected): Evaluate performance of the classifier under three conditions: without augmentation, with rotation and with pitch shifting.
  
- PI5 (advanced): Find out for which chords the augmentation is especially beneficial (e.g. in rare keys), conduct per-chord analysis.

Objective 3: Improve the entire system by adding smoothing and learned features

- PI1 (basic): Propagate output probabilities of the CNN to the hidden Markov Model.
  
- PI2 (basic): Generate visualization of the chord timeline for the representative song both before and after smoothing.
  
- PI3 (expected): Quantify performance gains of CNN due to the smoothing.
  
- PI4 (expected): Use pitchclass_cnn features through the classifier and compare performance of such network with the performance of the one that takes CQT as input.
  
- PI5 (advanced): Train a convolutional recurrent neural network (CRNN) and compare the intrinsic smoothing of CRNN with CNN + HMM combination.

# References
- [1] M. McVicar, R. Santos-Rodríguez, Y. Ni, and T. De Bie, "Automatic chord estimation from audio: A review of the state of the art," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 22, no. 2, pp. 556–575, 2014.
  
- [2] J. Pauwels, K. O'Hanlon, E. Gómez, and M. B. Sandler, "20 years of automatic chord recognition from audio," in Proc. ISMIR, 2019.
  
- [3] E. J. Humphrey and J. P. Bello, "Rethinking automatic chord recognition with convolutional neural networks," in Proc. ICMLA, 2012.
  
- [4] F. Korzeniowski and G. Widmer, "Feature learning for chord recognition: The deep chroma extractor," in Proc. ISMIR, 2016.
  
- [5] C. Weiß, J. Zeitler, T. Zunner, F. Schuberth, and M. Müller, "Learning pitch-class representations from score–audio pairs of classical music," in Proc. ISMIR, 2021.
  
- [6]  Kristi311, and Alexey Gruzdev. “Krist311/Chords-Recognition.” GitHub, 2019, github.com/krist311/chords-recognition.
  
- [7] Maludi, Kurnia, et al. “Chord Identification Using Pitch Class Profile Method with Fast Fourier Transform Feature Extraction.” IJCSI International Journal of Computer Science Issues, vol. 11, no. 3, May 2014, pp. 139–144. 1.
  
- [8] HOUSSINEAU, JEREMIE, et al. “On Large Lag Smoothing for Hidden Markov Models.” ArXiv.Org, 19 Apr. 2018, arxiv.org/html/1804.07117v1.
  
- [9] Gunderson, Gregory. “Inference for Hidden Markov Models.” Gregorygunderson.Com, 28 Nov. 2020, https://gregorygundersen.com/blog/2020/11/28/hmms/. Accessed 16 Oct. 2026.
  
- [10] Cercone, Nick. “Hidden-Markov Model.” EECS6339. https://wiki.eecs.yorku.ca/course_archive/2014-15/W/6339/_media/lecture_19_cl_-_hidden_markov_models.pdf. Accessed 16 Oct. 2026.
  
- [11] “2025:Audio Chord Estimation,” 2025:Audio Chord Estimation - MIREX Wiki, https://music-ir.org/mirex/wiki/2025:Audio_Chord_Estimation (accessed Oct. 19, 2026).
  
- [12] Junnosuke Mimura and Tetsuya Mizutani, A fine-tuning method for automatic chord estimation using the Foundation Model | Proceedings of the 8th International Conference on Computer Science and Application Engineering, https://dl.acm.org/doi/10.1145/3704814.3704833 (accessed Oct. 19, 2026).
  


