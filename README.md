# PersonalGPT - Conversational Agent

PersonalGPT is a robust conversational agent named "Althea Helper" designed to assist users in a professional context. It orchestrates complex interactions, maintains conversation stages, and utilizes various tools to engage in meaningful and context-aware dialogues.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Workflow](#workflow)
5. [Contributing](#contributing)
6. [License](#license)
7. [Functionality](#functionality)

## Requirements

### Tools Required

- Python 3.x
- OpenAI GPT-3
- Pydantic
- Langchain

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Chat_ERP.git
   ```

2. **Navigate to the Directory**:
   ```bash
   cd personalgpt
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can interact with the PersonalGPT conversational agent by running the main script:

```bash
python personalgpt.py
```

You can interact with the web based dialogue system using:

```bash
python app.py
```

Follow the prompts to engage in a conversation with the agent.

## Workflow

We use a personal agent which is configured to run through stages of conversation to help a user through their tasks. These stages are Introduction, Identification, Suggestion, Execution and Completion. They help ChatGPT understand how to approach a conversation given a specific chat history.

The personal agent, "Althea Helper," is equipped with a set of specialized tools designed to facilitate various actions within the conversation. These tools are categorized into three main groups:

1. **TaskRelated**: Enables the agent to manage tasks, such as creating, updating, deleting, or retrieving information about specific tasks. It helps in organizing and tracking activities within a project or workflow.
2. **DataRelated**: Provides access to data-related operations, such as querying databases, manipulating data sets, or retrieving specific information. It serves as a bridge between the agent and the underlying data resources.
3. **Calculator**: Offers mathematical computation capabilities, allowing the agent to perform calculations, solve mathematical problems, and provide numerical analysis. It enhances the agent's ability to assist with numerical tasks and inquiries.

These tools augment the agent's core conversational abilities, allowing it to interact with the user in more meaningful ways, provide context-aware responses, and perform specific actions that go beyond mere dialogue. They contribute to a more interactive and dynamic conversational experience, tailored to the needs of the user. Problems with their determinism are explored in the [Functionality](#functionality) section.

The overall workflow can be seen in the following steps:
1. **Initialization**: Configure employee details and initialize conversation stage.
2. **User Interaction**: Accept user input and process it through the `human_step` method.
3. **Conversation Stage Determination**: Determine and update the current conversation stage.
4. **Agent Response**: Generate the agent's response using appropriate tools and conversation guidelines.
5. **Continuous Interaction**: Loop for ongoing conversation.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Functionality

While GPT is a traditionally non determinisitic machine, we have tried to make it as accurate as possible with smart prompting techniques.
The following prompts show good accuracy:

**Mail**
- [ ] Send mail - 
    - [ ] send a mail to xyz@gmail.com saying “—“ with the subject “—“
    - [ ] Send a mail to xyz@gmail.com conveying “—-“
- [ ] Draft mail - 
    - [ ] Same as send, prefer this to check errors!!
- [ ] Get draft - 
    - [ ] What is the last mail in my drafts?
    - [ ] What is the top mail in my drafts?
- [ ] Get mail - 
    - [ ] Find the last mail in my inbox from xyz@gmail.com

**Task**
- [ ] List Tasks
    - [ ] What are my tasks today?
    - [ ] Can you list my tasks for today?
- [ ] Add Tasks
    - [ ] Add a task with the description of XYZ and a priority of X
    - [ ] Add a task for XYZ/for the topic XYZ
    - [ ] Add a task to achieve XYZ
- [ ] Delete Tasks
    - [ ] Remove the task with the description of XYZ
    - [ ] Remove this task/the recent task

**Data**

