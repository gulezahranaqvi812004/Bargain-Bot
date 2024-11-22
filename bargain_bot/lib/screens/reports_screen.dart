import 'package:flutter/material.dart';

class ReportsScreen extends StatelessWidget {
  // Sample data for the report
  final List<Map<String, String>> negotiationReports = [
    {
      'productName': 'Smartphone',
      'originalPrice': '\$999',
      'userOffer': '\$850',
      'finalPrice': '\$890',
      'status': 'Accepted',
      'date': '2024-11-22 10:15',
    },
    {
      'productName': 'Laptop',
      'originalPrice': '\$1500',
      'userOffer': '\$1200',
      'finalPrice': '\$1250',
      'status': 'Rejected',
      'date': '2024-11-21 14:30',
    },
    // Add more reports here...
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Negotiation Reports"),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Report Title and Instructions
            Text(
              'Negotiation Report Summary',
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 16),
            Text(
              'Here are the details of your past negotiations. Tap on any negotiation to view more details.',
              style: TextStyle(fontSize: 16, color: Colors.grey[700]),
            ),
            SizedBox(height: 16),

            // Negotiation Report List
            Expanded(
              child: ListView.builder(
                itemCount: negotiationReports.length,
                itemBuilder: (context, index) {
                  var report = negotiationReports[index];
                  return Card(
                    margin: EdgeInsets.symmetric(vertical: 8.0),
                    elevation: 4.0,
                    child: ListTile(
                      title: Text(report['productName']!),
                      subtitle: Text('Negotiated Price: ${report['finalPrice']}'),
                      trailing: Text(report['status']!),
                      onTap: () {
                        // Navigate to a detailed view of the selected report
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => NegotiationDetailScreen(report: report),
                          ),
                        );
                      },
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class NegotiationDetailScreen extends StatelessWidget {
  final Map<String, String> report;

  const NegotiationDetailScreen({required this.report});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Negotiation Details"),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Product: ${report['productName']}',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 16),
            Text(
              'Original Price: ${report['originalPrice']}',
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 8),
            Text(
              'User Offer: ${report['userOffer']}',
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 8),
            Text(
              'Negotiated Price: ${report['finalPrice']}',
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 8),
            Text(
              'Negotiation Status: ${report['status']}',
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 8),
            Text(
              'Date: ${report['date']}',
              style: TextStyle(fontSize: 16),
            ),
            Spacer(),
            // Option to export/share the report
            ElevatedButton(
              onPressed: () {
                // Logic to export/share the report
                ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                  content: Text('Report shared or exported.'),
                ));
              },
              child: Text('Export/Share Report'),
            ),
          ],
        ),
      ),
    );
  }
}
