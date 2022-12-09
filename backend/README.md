# Backend

The backend fulfills the following tasks:

- Get ensemble dataset. For instance ellipse vs parotid slices vs weather.
- Processing dataset members or subsets. For instance, computing DICE between members or getting a contour from a grid
  or vice-versa.
- Computing method-specific data. For instance, the median, bands and outliers for contour boxplots.
- Creating sessions to store user interactions.
- Submission and storage of survey data and their related data.

Decoupling the backend in this way will allow us to implement purpose-dependent GUIs.
For instance, we can build a GUI to explore a particular method, another to compare several methods and another to
perform a user study.

