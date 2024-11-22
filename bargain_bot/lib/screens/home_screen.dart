import 'package:bargain_bot/screens/negotiation_screen.dart';
import 'package:bargain_bot/screens/reports_screen.dart';
import 'package:flutter/material.dart';

class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("BargainBot"),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Banner Section
            Container(
              padding: EdgeInsets.all(16.0),
              decoration: BoxDecoration(
                color: Colors.blue.shade100,
                borderRadius: BorderRadius.circular(12.0),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Text(
                    "Welcome to BargainBot!",
                    style: TextStyle(
                      fontSize: 24.0,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 8.0),
                  Text(
                    "Your AI-powered negotiation assistant for better deals.",
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 16.0, color: Colors.grey[700]),
                  ),
                ],
              ),
            ),
            SizedBox(height: 24.0),

            // Navigation Buttons
            HomeButton(
              icon: Icons.chat_bubble_outline,
              title: "Start Negotiation",
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => NegotiationScreen()),
                );
              },
            ),
            // SizedBox(height: 16.0),
            // HomeButton(
            //   icon: Icons.analytics_outlined,
            //   title: "Scope & Complexity Analysis",
            //   onTap: () {
            //     // Navigate to Scope Analysis Screen
            //   },
            // ),
            // SizedBox(height: 16.0),
            // HomeButton(
            //   icon: Icons.monetization_on_outlined,
            //   title: "Budget Estimation",
            //   onTap: () {
            //     // Navigate to Budget Estimation Screen
            //   },
            // ),
            // SizedBox(height: 16.0),
            // HomeButton(
            //   icon: Icons.api_outlined,
            //   title: "API Integration",
            //   onTap: () {
            //     // Navigate to API Integration Screen
            //   },
            // ),
            SizedBox(height: 16.0),
            HomeButton(
              icon: Icons.bar_chart_outlined,
              title: "Reports",
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => ReportsScreen()),
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}

class HomeButton extends StatelessWidget {
  final IconData icon;
  final String title;
  final VoidCallback onTap;

  const HomeButton({
    required this.icon,
    required this.title,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: EdgeInsets.all(16.0),
        decoration: BoxDecoration(
          color: Colors.blue.shade50,
          borderRadius: BorderRadius.circular(12.0),
          border: Border.all(color: Colors.blue, width: 1.5),
        ),
        child: Row(
          children: [
            Icon(icon, color: Colors.blue, size: 30.0),
            SizedBox(width: 16.0),
            Expanded(
              child: Text(
                title,
                style: TextStyle(fontSize: 18.0, fontWeight: FontWeight.w500),
              ),
            ),
            Icon(Icons.arrow_forward_ios, color: Colors.blue, size: 20.0),
          ],
        ),
      ),
    );
  }
}
