# 📱 Social Media & Mental Health Analysis

## 3D Interactive Dashboard | Male vs Female Behavior | Mental Health Impact

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Three.js](https://img.shields.io/badge/Three.js-r128-green)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## 🧠 Project Overview

This project analyzes the relationship between **social media usage** and **mental health** (anxiety & depression), comparing behavioral patterns between **Males** and **Females**.

### Problem Statement
> Excessive screen time and prolonged engagement on social media platforms are increasingly associated with anxiety, depression, and other mental health challenges. The impact differs across gender groups due to differences in digital behavior, emotional expression, and coping mechanisms.

### Key Questions Answered
- What are males watching on social media?
- What are females watching on social media?
- How many hours does each gender spend?
- How does screen time affect mental health?

---

## ✨ Features

### 🎨 3D Visual Effects
- **Particle Background System** - Floating particles responding to mouse movement
- **3D Tilt Cards** - Interactive cards with depth effect on hover
- **Rotatable 3D Charts** - Drag to view from any angle
- **Animated Risk Gauges** - Sweeping needle animation showing risk percentages

### 📊 Interactive Data Visualization
| Chart Type | Description |
|------------|-------------|
| 3D Bar Chart | Screen time & risk comparison by gender |
| 3D Scatter Plot | GAD-7 vs PHQ-9 correlation |
| 3D Surface Heatmap | Risk mapping based on behavior |
| 3D Pie Chart | Platform distribution with exploding segments |
| 3D Carousel | Platform usage with click details |

### 📈 Key Metrics Tracked
- Screen Time (hours/day)
- GAD-7 Score (Anxiety)
- PHQ-9 Score (Depression)
- Late Night Usage (%)
- Social Comparison Trigger (%)
- High Risk Users (%)

---

## 🚀 Live Demo

Open `index.html` in any modern browser - No server required!

```bash
# Just double-click the file
open index.html

📊 Data Source
Dataset: Social Media & Mental Health Survey
Total Records: 7,163 users
Features: 16 attributes including:

Demographics (Age, Gender)

Usage patterns (Screen Time, Late Night, Platform)

Mental health scores (GAD-7, PHQ-9)

Behavioral triggers (Social Comparison, Activity Type)

🎯 Key Findings
Male Behavior
Metric	Value
Avg Screen Time	5.34 hrs/day
Top Platform	YouTube
Top Content	Gaming
Late Night Usage	42%
Social Comparison	28%
High Risk	27.5%
Female Behavior
Metric	Value
Avg Screen Time	5.12 hrs/day
Top Platform	Instagram
Top Content	Lifestyle/Beauty
Late Night Usage	39%
Social Comparison	56%
High Risk	28.9%
Gender Comparison
📱 Screen Time: Males +0.22 hrs more

🧠 Social Comparison: Females +27.7% higher

😟 Anxiety: Females score higher (7.23 vs 6.94)

😔 Depression: Females score higher (7.72 vs 7.39)

🛠️ Technologies Used
Technology	Purpose
Three.js	3D particle background, carousel
Plotly.js	3D interactive charts
GSAP	Scroll animations, counters
Vanilla Tilt	3D card tilt effects
FontAwesome	Icons
HTML5/CSS3	Structure & styling
📁 Project Structure
social-media-mental-health/
├── index.html          # Complete 3D dashboard (single file)
├── README.md           # Documentation
└── assets/             # (Optional) Images for README

🖥️ Browser Compatibility
Browser	Version	Status
Chrome	90+	✅ Fully Supported
Firefox	88+	✅ Fully Supported
Edge	90+	✅ Fully Supported
Safari	14+	✅ Supported
Mobile Chrome	Latest	✅ Responsive

🔧 Installation
# Clone the repository
git clone https://github.com/yourusername/social-media-mental-health.git

# Navigate to project folder
cd social-media-mental-health

# Open in browser (no build required)
open index.html

🎮 How to Use
Hover over cards - See 3D tilt effect

Click on gender cards - Flip to see detailed statistics

Drag 3D charts - Rotate to view from any angle

Click platform cards - See detailed impact analysis

Scroll down - Animated counters and gauges activate

📸 Screenshots
Hero Section
https://via.placeholder.com/800x400?text=Hero+Section+with+3D+Background

Gender Comparison Cards
https://via.placeholder.com/800x400?text=3D+Gender+Cards

3D Charts
https://via.placeholder.com/800x400?text=Interactive+3D+Charts

Platform Carousel
https://via.placeholder.com/800x400?text=3D+Platform+Carousel

💡 Recommendations
For Females
Reduce social comparison content consumption

Limit late night scrolling

Follow educational content over lifestyle content

For Males
Limit gaming sessions after 11 PM

Take regular breaks between gaming

Balance screen time with physical activity

For Both Genders
No phone 1 hour before sleep

Use screen time tracking apps

Follow positive, educational content

🔮 Future Enhancements
Real-time data API integration

User login for personalized tracking

Mobile app version

Weekly mental health report generation

AI-powered recommendation engine

Export data as PDF/CSV

👥 Team
Name	Role
Mahasri K	ML Lead & Backend
Rohith A	Frontend & 3D Visualization
Suhitha M	Data Analysis & Documentation

📄 License
MIT License - Free for academic and research use

🙏 Acknowledgments
Dataset: Kaggle Social Media & Mental Health Survey

Three.js community for 3D visualization resources

Plotly for interactive charting libraries

📧 Contact
For questions or collaboration:

GitHub Issues: Create an issue

Email: project@example.com

⭐ Star This Project
If you found this useful, please star the repository!
git star https://github.com/yourusername/social-media-mental-health

Made with 🧠 for better mental health awareness

---

## Also Create `.gitignore` File

```gitignore
# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE files
.vscode/
.idea/
*.swp
*.swo

# Temporary files
*.tmp
*.log
*.cache

# Backup files
*.bak
*.backup

# Local config
.env
local.properties

GitHub Upload Steps:
# Step 1: Initialize git
git init

# Step 2: Add files
git add index.html
git add README.md
git add .gitignore

# Step 3: Commit
git commit -m "Initial commit: 3D Social Media Mental Health Dashboard"

# Step 4: Create repository on GitHub
# (Do this on github.com - click New Repository)

# Step 5: Connect and push
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main


