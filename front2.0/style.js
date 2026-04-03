// Enhanced Chart for Modern UI
const canvas = document.getElementById("trend");
const ctx = canvas.getContext("2d");

// Set canvas size
canvas.width = canvas.parentElement.clientWidth;
canvas.height = 180;

// Chart data
const data = [45, 65, 80, 70, 120, 95, 140, 110, 85, 125, 150, 130];
const labels = ['8AM', '9AM', '10AM', '11AM', '12PM', '1PM', '2PM', '3PM', '4PM', '5PM', '6PM', '7PM'];

// Calculate points
const padding = 40;
const chartWidth = canvas.width - padding * 2;
const chartHeight = canvas.height - padding * 2;
const maxVal = Math.max(...data);
const minVal = Math.min(...data);
const range = maxVal - minVal;

// Draw grid lines
ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
ctx.lineWidth = 1;

for (let i = 0; i <= 4; i++) {
    const y = padding + (chartHeight / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(canvas.width - padding, y);
    ctx.stroke();
}

// Create gradient for area fill
const gradient = ctx.createLinearGradient(0, padding, 0, canvas.height - padding);
gradient.addColorStop(0, 'rgba(99, 102, 241, 0.3)');
gradient.addColorStop(0.5, 'rgba(139, 92, 246, 0.2)');
gradient.addColorStop(1, 'rgba(6, 182, 212, 0)');

// Draw area fill
ctx.beginPath();
ctx.moveTo(padding, canvas.height - padding);

data.forEach((val, i) => {
    const x = padding + (chartWidth / (data.length - 1)) * i;
    const y = canvas.height - padding - ((val - minVal) / range) * chartHeight;
    ctx.lineTo(x, y);
});

ctx.lineTo(canvas.width - padding, canvas.height - padding);
ctx.closePath();
ctx.fillStyle = gradient;
ctx.fill();

// Draw line
ctx.beginPath();
ctx.moveTo(padding, canvas.height - padding - ((data[0] - minVal) / range) * chartHeight);

data.forEach((val, i) => {
    const x = padding + (chartWidth / (data.length - 1)) * i;
    const y = canvas.height - padding - ((val - minVal) / range) * chartHeight;
    ctx.lineTo(x, y);
});

ctx.strokeStyle = '#6366f1';
ctx.lineWidth = 3;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.stroke();

// Glow effect
ctx.shadowColor = '#6366f1';
ctx.shadowBlur = 15;
ctx.stroke();
ctx.shadowBlur = 0;

// Draw data points
data.forEach((val, i) => {
    const x = padding + (chartWidth / (data.length - 1)) * i;
    const y = canvas.height - padding - ((val - minVal) / range) * chartHeight;
    
    // Outer glow
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(99, 102, 241, 0.3)';
    ctx.fill();
    
    // Inner point
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#6366f1';
    ctx.fill();
    
    // White center
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
});

// Draw labels
ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
ctx.font = '11px Inter, sans-serif';
ctx.textAlign = 'center';

labels.forEach((label, i) => {
    const x = padding + (chartWidth / (data.length - 1)) * i;
    ctx.fillText(label, x, canvas.height - 10);
});

// Add animation on load
canvas.style.opacity = '0';
canvas.style.transform = 'translateY(20px)';
canvas.style.transition = 'all 0.6s ease-out';

setTimeout(() => {
    canvas.style.opacity = '1';
    canvas.style.transform = 'translateY(0)';
}, 100);

// Simulated real-time updates
function updateStats() {
    // Add subtle animation to stat numbers
    const statNumbers = document.querySelectorAll('.stat-card h1');
    statNumbers.forEach(num => {
        num.style.transform = 'scale(1.05)';
        setTimeout(() => {
            num.style.transform = 'scale(1)';
        }, 200);
    });
}


setInterval(updateStats, 3000);
