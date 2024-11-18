let currentMode = 'drawing';
let color = '#000000';
let thickness = 5;

document.getElementById('colorPicker').addEventListener('input', (event) => {
    color = event.target.value;
    fetch(`/set_color?color=${encodeURIComponent(color)}`);
});

document.getElementById('thickness').addEventListener('input', (event) => {
    thickness = event.target.value;
    fetch(`/set_thickness?thickness=${encodeURIComponent(thickness)}`);
});
function setMode(mode) {
    currentMode = mode;
    document.getElementById('modeIndicator').textContent = `Mode: ${mode.charAt(0).toUpperCase() + mode.slice(1)}`;
    fetch(`/set_mode/${mode}`);
}

function clearCanvas() {
    fetch('/clear')
        .then(response => console.log('Canvas cleared'))
        .catch(error => console.error('Error clearing canvas:', error));
}

function showSaveOptions() {
    document.getElementById('overlay').classList.add('active');
    document.getElementById('saveOptions').classList.add('active');
}

function closeSaveOptions() {
    document.getElementById('overlay').classList.remove('active');
    document.getElementById('saveOptions').classList.remove('active');
}

function saveAsImage() {
    fetch('/save_image')
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'canvas.png';
            a.click();
            window.URL.revokeObjectURL(url);
            closeSaveOptions();
        });
}

function saveAsPDF() {
    fetch('/save_pdf')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.blob();
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'canvas.pdf';
            a.click();
            window.URL.revokeObjectURL(url);
        })
        .catch(error => console.error('Error saving PDF:', error));
}
function saveAsImage() {
    fetch('/save_image')
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'canvas.png';
            a.click();
            window.URL.revokeObjectURL(url);
            closeSaveOptions();
        })
        .catch(error => console.error('Error saving image:', error));
}