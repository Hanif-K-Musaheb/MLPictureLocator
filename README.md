# Machine Learning Picture Locator
#### [Project short proposal](https://docs.google.com/document/d/1KCMH6d2X6r1-BFHSYg7oauPKJSveUPxbLv-iIyglzGM/edit?tab=t.0)
#### [GSV database](https://www.kaggle.com/datasets/amaralibey/gsv-cities)
------

### What we need to do:
1. **Data Prep**:
- [x] find a way to get the city out of the image name
- [x] create a function to split the data of every city into a 10% test, 10% validation, 80% training
- [x] data loader
- [x] resize images
2. **Building the Architecture**:
- [ ] get the CNN to image classify
3. **The Training Loop**
- [ ]  Forward Pass: The computer takes an image, passes it through the network, and makes a guess (e.g., "I am 80% sure this is Chicago")
- [ ]  Calculate Loss: The computer checks the actual folder name. If it was actually Boston, a loss function (a mathematical formula that calculates exactly how wrong the computer's guess was) generates an error score.
- [ ]  Backpropagation: The computer works backward through its math, adjusting its internal weights (the numbers inside the network that determine how important certain features are) to make sure it guesses closer to "Boston" next time.
- [ ] repeat
