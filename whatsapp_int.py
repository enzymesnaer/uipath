from twilio.rest import Client

account_sid = 'AC03e77649f31d37eb2d902d3e4c906a97'
auth_token = '61462ebc9c022ab5a11f2ded13a0f24b'

def send_whatsapp(output_file):
    client = Client(account_sid, auth_token)
    numbers = ['whatsapp:+918285281211']
    
    with open('output/audio_summary.txt') as f:
        lines=f.readlines()
    file1= ''.join(lines)[:1598]

    for number in numbers:
        message = client.messages.create(
        from_='whatsapp:+14155238886',
        body = file1,
        to   = number
        )

if __name__ == "__main__":
    text_regex = re.compile(r'.*\.txt$', re.IGNORECASE)
    txt_files = [file for file in glob.glob('C:\\Users\\kolisn\\Documents\\Workspace\\hackathon\\MVP\\dropbox\\' + '*')
                 if text_regex.match(file)]

    send_whatsapp(text_file)


print(message.sid)