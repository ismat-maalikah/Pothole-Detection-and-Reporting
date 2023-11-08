# Pothole-Detection-and-Reporting
Using CNN and deep learning techniques, the detection model is trained and optimized. The sequential deep learning method is used to train the model. Hyper-parameter tuning using 
Keras Bayesian Optimization Tuner is used to get efficient results and for the model to consider the best optimization model through epochs.
The detection model gives 91% accuracy.
Not only do the potholes need to be detected they also need to be reported. 
Potholes can be reported to the authorities through SMS or an EMAIL. The current coordinates of the pothole will be sent to the authorities and they can take the necessary action for reducing
potholes on roads.
For reporting via SMS, Twillio API integration is done. PHP is used to include functions and configurations for the Twilio service.
This PHP script essentially processes the form data to send an SMS. 
The combined HTML and PHP code creates a form that allows users to report potholes via email and demonstrates the usage of the PHPMailer library for sending emails. 
This PHP code integrates the form data with the PHPMailer library to send an email report about a pothole, including various user-provided details and an image attachment. 
It also handles potential errors during the email-sending process.
