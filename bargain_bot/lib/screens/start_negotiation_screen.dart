import 'package:flutter/material.dart';

class StartNegotiationScreen extends StatefulWidget {
  @override
  _StartNegotiationScreenState createState() => _StartNegotiationScreenState();
}

class _StartNegotiationScreenState extends State<StartNegotiationScreen> {
  String negotiationStatus = "Negotiation is about to begin...";
  String botResponse = "Please wait while I negotiate with the seller...";
  String userResponse = ""; // To store user's response

  // Function to simulate negotiation
  void startNegotiation() {
    setState(() {
      negotiationStatus = "Negotiation in progress...";
      botResponse = "I'm negotiating the price...";
    });

    // Simulate some negotiation process
    Future.delayed(Duration(seconds: 3), () {
      setState(() {
        negotiationStatus = "Negotiation Completed!";
        botResponse = "You got a great deal! Your final price is \$85.";
      });
    });
  }

  // Function to save user's response (Accept or Reject)
  void saveResponse(String response) {
    setState(() {
      userResponse = response; // Save the user's response
    });

    // Display a confirmation message
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('You have $response the offer.'),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Negotiation in Progress"),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Status and Bot Response
            Text(
              negotiationStatus,
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Colors.blue,
              ),
            ),
            SizedBox(height: 16),
            Text(
              botResponse,
              style: TextStyle(
                fontSize: 16,
                color: Colors.grey[700],
              ),
            ),
            SizedBox(height: 32),

            // Start Negotiation Button
            ElevatedButton(
              onPressed: startNegotiation,
              child: Text(
                'Start Negotiation',
                style: TextStyle(fontSize: 16),
              ),
              style: ElevatedButton.styleFrom(
                padding: EdgeInsets.symmetric(vertical: 14.0),
              ),
            ),
            SizedBox(height: 32),

            // Accept or Reject Offer Buttons
            if (negotiationStatus == "Negotiation Completed!") ...[
              Text(
                'Do you accept the offer?',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                    onPressed: () => saveResponse("accepted"),
                    child: Text("Accept"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.green,
                      padding: EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                    ),
                  ),
                  SizedBox(width: 16),
                  ElevatedButton(
                    onPressed: () => saveResponse("rejected"),
                    child: Text("Reject"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.red,
                      padding: EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                    ),
                  ),
                ],
              ),
            ],

            SizedBox(height: 32),

            // Display user's response
            if (userResponse.isNotEmpty) ...[
              Text(
                'You have $userResponse the offer.',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: Colors.blue,
                ),
              ),
            ],
            Spacer(),
          ],
        ),
      ),
    );
  }
}
